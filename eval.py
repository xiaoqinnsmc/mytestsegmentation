from dataset import LiTSDataset
import transforms as T
import torch
import torch.utils.tensorboard
import os
import argparse
import time
import utils
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import utils
import numpy as np


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    """使用voc数据集评估

    Parameters
    ----------
    model : Module
        用于评估的模型
    data_loader : Dataloader
        用于读取验证集的Dataloader
    device : Device
        所使用的设备
    args : 
        命令行参数
    """
    # n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    DSCs, VOEs, RVDs, ASDs, MSDs, RMSDs, PAs = [], [], [], [], [], [], []

    for images, targets in metric_logger.log_every(data_loader, 100, 'Test:'):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pt_masks = output['masks'].to(cpu_device)
            pt_labels = output['labels'].to(cpu_device)
            pt_masks = [pt_masks[pt_labels == l] > 0.4 for l in torch.unique(pt_labels)]
            pt_masks = [torch.any(m, dim=0) for m in pt_masks]

            gt_masks = target['masks']
            # for m in gt_masks:
            #     assert torch.any(m).item()
            gt_labels = target['labels']
            gt_masks = [gt_masks[gt_labels == l] > 0.4 for l in torch.unique(gt_labels)]
            gt_masks = [torch.any(m, dim=0) for m in gt_masks]

            DSCs.extend([utils.dice_coef(pt_mask, gt_mask) for pt_mask, gt_mask in zip(pt_masks, gt_masks)])
            VOEs.extend([utils.voe_coef(pt_mask, gt_mask) for pt_mask, gt_mask in zip(pt_masks, gt_masks)])
            RVDs.extend([utils.rvd_coef(pt_mask, gt_mask) for pt_mask, gt_mask in zip(pt_masks, gt_masks)])
            ASDs.extend([utils.asd_coef(pt_mask, gt_mask) for pt_mask, gt_mask in zip(pt_masks, gt_masks)])
            MSDs.extend([utils.msd_coef(pt_mask, gt_mask) for pt_mask, gt_mask in zip(pt_masks, gt_masks)])
            RMSDs.extend([utils.rmsd_coef(pt_mask, gt_mask) for pt_mask, gt_mask in zip(pt_masks, gt_masks)])
            PAs.extend([utils.pixel_accuracy_class(gt_mask, pt_mask, num_class=2) for pt_mask, gt_mask in zip(pt_masks, gt_masks)])
        model_time = time.time() - model_time

        metric_logger.update(model_time=model_time)
    print("DSC:{:f}, VOE:{:f}, RVD:{:f}, ASD:{:f}, MSD:{:f}, EMSD:{:f}, PA:{:f}".format(
        np.mean(DSCs), np.mean(VOEs), np.mean(RVDs), np.mean(ASDs), np.mean(MSDs), np.mean(RMSDs), np.mean(PAs)
    ))


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--batch_size', type=int, default=8, help='default: {:g}'.format(8))
    parser.add_argument('--num_steps_to_display', type=int, default=20, help='default: {:d}'.format(20))
    parser.add_argument('--workers', type=int, default=4, help='default: {:d}'.format(4))
    parser.add_argument('--intra_class', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='coco2017', help='default: {:s}'.format('coco2017'))
    args = parser.parse_args()

    args.num_categories = 3

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # number of categories includes background
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=args.num_categories)
    # move model to the right device
    model.to(device)

    checkpoint_path = os.path.join(args.model_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['segmenter'])
        print('checkpoint loaded.')
    else:
        print('{:s} does not exist.'.format(checkpoint_path))

    # evaluate on the test dataset

    ct_transform = T.Compose([T.ToTensor()])
    dataset = LiTSDataset(args.dataset_dir, ct_transform)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        collate_fn=collate_fn)

    evaluate(model, data_loader, device, args)
