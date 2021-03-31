from os import truncate
import random
import torch
import numpy as np
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if "bbox" in target:
                bbox = target["bbox"]
                # assert torch.all(bbox[:, 2:] > bbox[:, :2])
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["bbox"] = bbox

            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Crop(object):
    def __init__(self, bound):
        self.bound = bound  # xmin, ymin, xmax, ymax

    def __call__(self, image, target):
        boxes = target['boxes']  # xmin, ymin, xmax, ymax
        boxes[:, 0] = torch.clamp_min(boxes[:, 0], self.bound[0]) - self.bound[0]
        boxes[:, 1] = torch.clamp_min(boxes[:, 1], self.bound[1]) - self.bound[1]
        boxes[:, 2] = torch.clamp(boxes[:, 2], self.bound[0], self.bound[2]) - self.bound[0]
        boxes[:, 3] = torch.clamp(boxes[:, 3], self.bound[1], self.bound[3]) - self.bound[1]
        # boxes += 1

        target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        image = image[:, self.bound[1]:self.bound[3], self.bound[0]:self.bound[2]]
        assert torch.all(target['area'] > 0)
        return image, target


class COCOAnnotationCollate(object):
    """整合coco数据集中的字段，用于训练
    """
    def __call__(self, image, target):
        collated_target = {}
        if len(target) != 0:
            keys = target[0].keys()
            for key in keys:
                _list = [t[key] for t in target if not t['iscrowd']]
                assert len(_list) != 0

                try:
                    collated_target[key] = torch.as_tensor(_list)
                except ValueError:
                    collated_target[key] = _list
                except Exception as e:
                    raise e

        return image, collated_target

    @staticmethod
    def reverse(target):
        assert len(target) != 0
        reversed = [{
            k: v[i] for k, v in target.items()
        } for i in range(target.values()[0].size(0))]
        return reversed


class VOCAnnotationCollate(object):
    """整合voc数据集中的字段，用于训练
    """
    def __init__(self, secondary_index):
        self.secondary_index = secondary_index

    def __call__(self, image, target):
        filename = target['annotation']['filename'].replace('.jpg', '.xml')
        target = target['annotation']['object']
        collated_target = {}
        if len(target) != 0:
            keys = target[0].keys()
            for key in keys:
                if key in ['truncated', 'diffcult']:
                    _list = [int(t[key]) for t in target]
                elif key == 'bndbox':
                    _list = [[
                        int(t[key]['xmin']), int(t[key]['ymin']),
                        int(t[key]['xmax']), int(t[key]['ymax'])
                    ] for t in target]
                elif key == 'part':
                    continue
                else:
                    _list = [t[key] for t in target]
                assert len(_list) != 0

                try:
                    collated_target[key] = torch.as_tensor(_list)
                except ValueError:
                    collated_target[key] = _list
                except Exception as e:
                    raise e

            collated_target['filename'] = [filename]*len(_list)
            collated_target['id'] = torch.as_tensor([self.secondary_index[filename][i] for i in range(len(_list))])

        return image, collated_target

    @staticmethod
    def reverse(target):
        assert len(target) != 0
        reversed = [{
            k: v[i] for k, v in target.items()
        } for i in range(target.values()[0].size(0))]
        return reversed


class BoxesFormatConvert(object):
    """x,y,w,h转换为left,top,right,bottom
    """
    def __call__(self, image, target):
        try:
            bbox = target["bbox"]
            bbox = bbox[torch.all(bbox[:, 2:] > 1e-6, dim=-1)]  # filter illegal labels
            bbox[:, 2:] += bbox[:, :2]
            target["bbox"] = bbox
        except KeyError:
            pass

        return image, target


class ClassConvert(object):
    """将每个标签转换为对应的子类别，或是将类别ID做转换（连续数组）
    """
    def __init__(self, cvt_map, num_intra_list=None):
        self.cvt_map = cvt_map
        self.num_intra_list = num_intra_list

    def __call__(self, image, target):
        try:
            if self.num_intra_list is not None:
                category_id = [self.cvt_map[i.item()] for i in target['id']]  # ann id to cat id
                target['category_id'] = torch.as_tensor(category_id, dtype=torch.int64)
            else:
                category_id = [self.cvt_map[i.item()] for i in target['category_id']]  # cat id to cat id
                target['category_id'] = torch.as_tensor(category_id, dtype=torch.int64)

        except KeyError:
            pass
        except Exception as e:
            raise e
        return image, target

    @staticmethod
    def _reverse_intra_classes(target, num_intra_list, cat_ids=None):
        if isinstance(target, dict):
            if len(target) == 0:
                return target
            labels = target['labels']
            for i, label in enumerate(labels):  # label start from 1
                for j, num_intra in enumerate(num_intra_list):
                    if label <= num_intra:
                        break
                    label -= num_intra
                if cat_ids:
                    labels[i] = cat_ids[j]
                else:
                    labels[i] = j + 1  # class indices start from 1
            target['labels'] = labels
        elif isinstance(target, int):
            label = target
            for j, num_intra in enumerate(num_intra_list):
                if label <= num_intra:
                    break
                label -= num_intra
            if cat_ids:
                target = cat_ids[j]
            else:
                target = j + 1  # class indices start from 1
        return target

    @staticmethod
    def _reverse_original_classes(target, cat_ids):
        assert cat_ids is not None, "cat_ids can't be None when using original category"
        if isinstance(target, dict):
            if len(target) == 0:
                return target
            labels = target['labels']
            for i, label in enumerate(labels):  # label start from 1
                labels[i] = cat_ids[label-1]
            target['labels'] = labels
        elif isinstance(target, int):
            target = cat_ids[target-1]  # class indices start from 1
        return target

    @staticmethod
    def reverse(target, num_intra_list, cat_ids=None):
        """Use after collate annotation

        Parameters
        ----------
        target : Dict
            Annotation after collating
        num_intra_list : List[int]
            Numbers of each intra class
        cat_ids : List[int], optional
            Dict used to convert cat ids to coco cat ids, by default None

        Returns
        -------
        Dict
            class indices start from 1
        """
        if num_intra_list is not None:
            return ClassConvert._reverse_intra_classes(target, num_intra_list, cat_ids)
        else:
            return ClassConvert._reverse_original_classes(target, cat_ids)

class CategoryToLabeledClasses(object):
    """类别转换为ID（连续数组）
    """
    def __init__(self, cvt_map):
        self.cvt_map = cvt_map

    def __call__(self, image, target):
        try:
            category_id = [self.cvt_map[i] for i in target['name']]  # ann id to cat id
            target['category_id'] = torch.as_tensor(category_id, dtype=torch.int64)

        except KeyError:
            pass
        except Exception as e:
            raise e
        return image, target
