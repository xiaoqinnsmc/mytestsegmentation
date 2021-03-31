import argparse
import os

import numpy as np
import torch
import glob
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

import json


class Detector(object):
    def __init__(self, args):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = args.num_classes
        self.checkpoint = torch.load(args.model_path, map_location=self.device)
        self.categories = args.categories
        self.model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=len(args.categories))

        # number of categories includes background
        self.model.load_state_dict(self.checkpoint['segmenter'])
        self.model.to(self.device)

        self._index = 1

    def visualize_mask(self, src, mask, color):
        import cv2
        src = np.array(src.convert('RGB'))
        mask = mask.cpu().numpy().squeeze() > 0.05
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(src, contours, -1, color, 1)  # index=-1表示画所有的contour

        return Image.fromarray(src)

    def from_global_image(self, global_image, thres=0.9):
        self.model.eval()
        img_input = TF.to_tensor(global_image)
        with torch.no_grad():
            prediction = self.model([img_input.to(self.device)])[0]

        draw = ImageDraw.Draw(global_image)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for score, box, label, mask in zip(prediction['scores'], prediction['boxes'], prediction['labels'], prediction['masks']):
            if score < thres:
                continue
            # assert score != 2
            xmin = round(box[0].item())
            ymin = round(box[1].item())
            xmax = round(box[2].item())
            ymax = round(box[3].item())
            category = self.categories[label - 1]
            # draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='red', width=3)
            draw.text((xmin, ymin), text=f'{category:s} {score.item():.3f}', fill='red')
            global_image = self.visualize_mask(global_image, mask, colors[label - 1])

        return global_image

    def save_as_labelme(self, output_dir, global_image, sub_region, nms_thres=0.9):
        image = global_image.crop(sub_region)
        scores, boxes, labels = self.detect(image, nms_thres)

        image_path = os.path.join(output_dir, "{:d}_real.jpg".format(self._index))
        content = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": image.height,
            "imageWidth": image.width
        }
        for i in range(labels.size(0)):
            box = boxes[i]
            label = labels[i].item()
            xmin = round(box[0].item())
            ymin = round(box[1].item())
            xmax = round(box[2].item())
            ymax = round(box[3].item())
            content["shapes"].append(
                {
                    "label": self.labels_cvt[label],
                    "points": [
                        [
                            xmin,
                            ymin
                        ],
                        [
                            xmax,
                            ymax
                        ]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            )
        json.dump(content, open(image_path.replace('.jpg', '.json'), 'w'))
        image.save(image_path)
        self._index += 1

    def load_labels(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.labels_cvt = {int(l.split(' ')[0]): l.split(' ')[1].rstrip() for l in lines}


def _infer_from_image():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-m', '--model_path', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='inputs directory')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='outputs directory')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    args = parser.parse_args()

    # load classes (including background)
    args.num_classes = 3
    args.categories = ['0', '1', '2']
    os.makedirs(args.output_dir, exist_ok=True)

    detector = Detector(args)

    for path in glob.glob(os.path.join(args.input_dir, '*.jpg')):
        global_image = Image.open(path)
        # if '000050' in path:
        #     print()
        global_image = detector.from_global_image(global_image, thres=0.6)
        global_image.save(os.path.join(args.output_dir, os.path.basename(path).replace('.jpg', '.png')))


def _infer_from_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./medicine_data', help='path to data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--num_classes', type=int, default=4, help='default: {:d}'.format(75))
    args = parser.parse_args()

    import cv2
    sub_region = [800, 150, 1300, 920]
    detector = Detector(args)
    cap = cv2.VideoCapture(args.input_path)
    while(cap.isOpened()):
        _, frame = cap.read()
        global_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        global_image = detector.from_global_image(global_image, sub_region, thres=0.6)
        global_image = global_image.resize(tuple(int(i*0.8) for i in global_image.size))
        cv2.imshow('image', cv2.cvtColor(np.array(global_image), cv2.COLOR_BGR2RGB))
        k = cv2.waitKey(20)
        # q键退出
        if k & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def _regen_labeled_images():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./medicine_data', help='path to data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='path to inputs')
    parser.add_argument('-o', '--output_dir', type=str, default='./regenerated_data', help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--num_classes', type=int, default=4, help='default: {:d}'.format(75))
    args = parser.parse_args()

    import cv2
    SUB_REGION = [800, 150, 1300, 920]
    SAMPLE_INTERVAL = 300
    detector = Detector(args)
    cap = cv2.VideoCapture(args.input_path)
    counter = 0
    while cap.isOpened():
        valid, frame = cap.read()
        if not valid:
            break
        if counter % SAMPLE_INTERVAL == 0:
            global_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detector.save_as_labelme(args.output_dir, global_image, SUB_REGION, nms_thres=0.6)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _infer_from_image()
    # _infer_from_video()
    # _regen_labeled_images()
