import os
import SimpleITK as sitk
import numpy as np
import json
from PIL import Image, ImageDraw


def mask2box(mask, cat):
    indices = np.where(mask == cat)
    ymin = int(indices[0].min()) - 1
    xmin = int(indices[1].min()) - 1
    ymax = int(indices[0].max()) + 1
    xmax = int(indices[1].max()) + 1

    return [xmin, ymin, xmax, ymax]


def visualize_segment(ct, seg):
    color_names = ['pink', 'blue', 'green', 'yellow']
    seg = np.array(seg)
    vis = ct.convert('RGBA')
    for cat in np.unique(seg)[1:]:
        color = Image.new('RGBA', vis.size, color_names[cat])
        color.putalpha(80)
        vis.paste(color, None, Image.fromarray(seg == cat))
    return vis


def visualize_boxes(ct, boxes):
    color_names = ['pink', 'blue', 'green', 'yellow']
    draw = ImageDraw.Draw(ct)

    for c, b in boxes.items():
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=color_names[c], width=2)

    return ct


ct_dir = 'data/CT'
seg_dir = 'data/seg'
out_ct_dir = 'data/final_CT'
out_seg_dir = 'data/final_seg'
out_ann_dir = 'data/final_ann'
out_vis_dir = 'data/visualization'

os.makedirs(out_ct_dir, exist_ok=True)
os.makedirs(out_seg_dir, exist_ok=True)
os.makedirs(out_ann_dir, exist_ok=True)
os.makedirs(out_vis_dir, exist_ok=True)

ct_list = os.listdir(ct_dir)
seg_list = map(lambda x: x.replace('volume', 'segmentation'), ct_list)

ct_list = map(lambda x: os.path.join(ct_dir, x), ct_list)
seg_list = map(lambda x: os.path.join(seg_dir, x), seg_list)

count = 0
for ct_path, seg_path in zip(ct_list, seg_list):
    # 将CT和金标准读入到内存中
    ct_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_path, sitk.sitkInt16))
    seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_path, sitk.sitkUInt8))

    for ct, seg in zip(ct_array, seg_array):
        ct = (ct-ct.min())*255/(ct.max()-ct.min())
        ct = ct.astype(np.uint8)
        ct = Image.fromarray(ct)
        categories = np.unique(seg)
        if len(categories) == 1:
            continue

        ann = {int(i): mask2box(seg, i) for i in categories[1:]}
        # seg = Image.fromarray(seg.copy())
        vis = visualize_segment(ct, seg)
        vis = visualize_boxes(vis, ann)

        ct.save(os.path.join(out_ct_dir, '{:06d}.jpg'.format(count)), lossless=True)
        json.dump(ann, open(os.path.join(out_ann_dir, '{:06d}.json'.format(count)), 'w'))
        assert np.all([np.any(seg==i) for i in categories[1:]])
        np.save(os.path.join(out_seg_dir, '{:06d}.npy'.format(count)), seg) 
        vis.save(os.path.join(out_vis_dir, '{:06d}.png'.format(count)))
        count += 1

    # import cv2
    # aa=ct_array[0]
    # aa=(aa-aa.min())*255/(aa.max()-aa.min())
    # cv2.imshow('1',aa.astype(np.uint8))
    # cv2.waitKey()
    # min max 归一化
