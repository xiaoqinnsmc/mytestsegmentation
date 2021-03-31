import torch
import os
from PIL import Image
from xml.dom.minidom import parse
import numpy as np
import glob
import json


class MarkDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

        with open(os.path.join(root, "labels.txt"), 'r') as f:
            lines = f.readlines()
            self.labels_cvtmap = {
                l.split(' ')[1].rstrip(): int(l.split(' ')[0]) for l in lines
            }

    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
        img = Image.open(img_path).convert("RGB")

        # 读取文件，VOC格式的数据集的标注是xml格式的文件
        dom = parse(bbox_xml_path)
        # 获取文档元素对象
        data = dom.documentElement
        # 获取 objects
        objects = data.getElementsByTagName('object')
        # get bounding box coordinates
        boxes = []
        labels = []
        for object_ in objects:
            # 获取标签中内容
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # 就是label，mark_type_1或mark_type_2
            labels.append(self.labels_cvtmap[name])  # 背景的label是0，mark_type_1和mark_type_2的label分别是1和2

            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # 由于训练的是目标检测网络，因此没有教程中的target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # 注意这里target(包括bbox)也转换\增强了，和from torchvision import的transforms的不同
            # https://github.com/pytorch/vision/tree/master/references/detection 的 transforms.py里就有RandomHorizontalFlip时target变换的示例
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class LiTSDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.ct_list = glob.glob(os.path.join(root, 'final_CT/*.jpg'))
        self.seg_list = glob.glob(os.path.join(root, 'final_seg/*.npy'))
        self.ann_list = glob.glob(os.path.join(root, 'final_ann/*.json'))

        self.transform = transform

    def __getitem__(self, index):
        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]
        ann_path = self.ann_list[index]

        # 将CT和金标准读入到内存中
        image_id, _ = os.path.splitext(os.path.basename(ct_path))
        ct = Image.open(ct_path)
        seg = np.load(seg_path)
        ann = json.load(open(ann_path, 'r'))

        ct, seg = self.transform(ct, seg)

        boxes, labels, masks, areas = [], [], [], []
        for c, b in ann.items():
            c = int(c)
            boxes.append(b)
            labels.append(c)
            mask = (c == seg)
            assert torch.any(mask).item(),seg_path+'{:d}'.format(c)
            masks.append(mask)
            areas.append(mask.sum())

        target = {}
        target["boxes"] = torch.tensor(boxes)
        target["labels"] = torch.tensor(labels)
        target["masks"] = torch.stack(masks)
        target["image_id"] = torch.tensor(int(image_id))
        # target["area"] = torch.tensor(areas)
        boxes = torch.tensor(boxes)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        # import cv2
        # aa=ct_array[0]
        # aa=(aa-aa.min())*255/(aa.max()-aa.min())
        # cv2.imshow('1',aa.astype(np.uint8))
        # cv2.waitKey()
        # min max 归一化

        return ct, target

    def __len__(self):
        return len(self.ct_list)
