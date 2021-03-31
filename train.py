import transforms as T
import torch
import torch.utils.tensorboard
import os
import argparse
import glob
from dataset import LiTSDataset
from lr_scheduler import WarmUpMultiStepLR
from torchvision.utils import save_image
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from model import Generator, Discriminator, get_adversarial_losses_fn, gradient_penalty
import transforms as T
import utils
import math
import sys
import functools
import torch.nn.functional as F
import copy

def collate_fn(batch):
    image, target = tuple(zip(*batch))
    image = torch.stack(image)
    target = list(target)
    return image, target


class Trainer(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        ct_transform = T.Compose([T.ToTensor()])
        dataset = LiTSDataset(args.dataset_dir, ct_transform)
        # define sampler
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)

        # define training and validation data loaders
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=args.workers,
            collate_fn=collate_fn)

        self.segmenter = maskrcnn_resnet50_fpn(pretrained=False, num_classes=args.num_categories)
        # move model to the right device
        self.segmenter.to(self.device)
        self.G = Generator(args.z_dim, args.x_dim).to(self.device)
        self.D = Discriminator(args.x_dim, n_downsamplings=args.n_D_downsamplings).to(self.device)
        # adversarial loss functions
        self.d_loss_fn, self.g_loss_fn = get_adversarial_losses_fn(args.adversarial_loss_mode)
        # SGD
        self.segmenter_optimizer = torch.optim.SGD(self.segmenter.parameters(), lr=args.learning_rate,
                                                   momentum=args.momentum, weight_decay=args.weight_decay)

        self.G_optimizer = torch.optim.SGD(self.G.parameters(), lr=args.learning_rate)
        self.D_optimizer = torch.optim.SGD(self.D.parameters(), lr=args.learning_rate)

        # and a learning rate scheduler
        self.lr_scheduler = WarmUpMultiStepLR(self.segmenter_optimizer, milestones=[60, 80], gamma=0.1,
                                              factor=0.3333, num_iters=5)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        self.summary_writer = torch.utils.tensorboard.SummaryWriter('logs')

        checkpoint_path = os.path.join(args.checkpoints_dir, 'checkpoint-epoch{:04d}.pth'.format(args.current_epoch-1))
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.segmenter_optimizer.load_state_dict(checkpoint['segmenter_optimizer'])
            self.segmenter.load_state_dict(checkpoint['segmenter'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print('checkpoint loaded.')
        else:
            print('{:s} does not exist.'.format(checkpoint_path))

    def visualize(self, images, masks, epoch):
        self.G.eval()
        images = torch.stack(images)
        x_fake = self.G(images, masks)
        # ma, _ = x_fake.max(dim=2, keepdim=True)
        # mi, _ = x_fake.min(dim=2, keepdim=True)
        # ma, _ = ma.max(dim=3, keepdim=True)
        # mi, _ = mi.min(dim=3, keepdim=True)

        # x_fake = (x_fake - mi) / (ma - mi)
        save_image(x_fake, 'logs/visualization/fakes-epoch-{:d}.jpg'.format(epoch), nrow=self.args.batch_size)
        save_image(images, 'logs/visualization/reals-epoch-{:d}.jpg'.format(epoch), nrow=self.args.batch_size)
        save_image(masks, 'logs/visualization/masks-epoch-{:d}.jpg'.format(epoch), nrow=self.args.batch_size)

    def train_G(self, images, masks, targets):
        self.G.train()
        self.D.train()
        images = torch.stack(images)

        x_fake = self.G(images, masks)

        x_fake_d_logit = self.D(x_fake)
        G_loss = self.g_loss_fn(x_fake_d_logit)

        for i in range(len(targets)):
            targets[i]['masks'] = masks[i]

        C_loss = F.mse_loss(x_fake, images)
        S_loss = sum(self.segmenter(x_fake, targets).values())
        loss = G_loss + C_loss + S_loss
        assert not torch.isnan(loss)
        self.G.zero_grad()
        loss.backward()
        self.G_optimizer.step()

        return {'g_loss': loss}

    def train_D(self, images, masks):
        self.G.train()
        self.D.train()
        images = torch.stack(images)

        images_fake = self.G(images, masks).detach()

        x_real_d_logit = self.D(images)
        x_fake_d_logit = self.D(images_fake)

        x_real_d_loss, x_fake_d_loss = self.d_loss_fn(x_real_d_logit, x_fake_d_logit)
        gp = gradient_penalty(
            functools.partial(self.D),
            images, images_fake,
            gp_mode=args.gradient_penalty_mode,
            sample_mode=args.gradient_penalty_sample_mode
        )

        D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

        self.D.zero_grad()
        D_loss.backward()
        self.D_optimizer.step()

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}

    def select_masks(self, masks):
        masks_indices = [torch.randint(mask.size(0), (1,)) for mask in masks]
        masks = torch.stack([masks[i][j] for i, j in enumerate(masks_indices)])
        return masks

    def train_one_epoch(self, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        batch_size = self.data_loader.batch_sampler.batch_size
        n_samples = len(self.data_loader)

        for step, (images, targets) in enumerate(metric_logger.log_every(self.data_loader, self.args.num_steps_to_display, header)):
            self.segmenter.train()

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items() if not isinstance(v, list)} for t in targets]
            _targets = copy.deepcopy( targets)

            masks = [t['masks'].to(torch.float32) for t in targets]
            masks = self.select_masks(masks)

            S_loss = self.segmenter(images, targets)

            self.segmenter_optimizer.zero_grad()
            assert not torch.isnan(sum(S_loss.values()))
            sum(S_loss.values()).backward()
            self.segmenter_optimizer.step()

            D_loss = self.train_D(images, masks)
            G_loss = self.train_G(images, masks, targets)

            losses = dict(S_loss, **D_loss, **G_loss)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(losses)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.segmenter_optimizer.param_groups[0]["lr"])

            metric_logger.to_summary_writer(
                self.summary_writer, 'training_step', epoch*(n_samples//batch_size)+step)

        metric_logger.to_summary_writer(self.summary_writer, 'training_epoch', epoch)
        self.visualize(images, masks, epoch)

        return metric_logger

    def snapshot(self, epoch):
        if epoch % self.args.num_epochs_to_snapshot == 0:
            os.makedirs('checkpoint', exist_ok=True)
            ckpts_path = glob.glob('checkpoint/checkpoint-epoch*.pth')
            if len(ckpts_path) > args.num_checkpoints_to_reserve:
                for path in ckpts_path[:len(ckpts_path)-args.num_checkpoints_to_reserve]:
                    os.remove(path)
            checkpoint = {
                'epoch': epoch,
                'segmenter_optimizer': self.segmenter_optimizer.state_dict(),
                'D': self.D.state_dict(),
                'G': self.G.state_dict(),
                'segmenter': self.segmenter.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
            }
            torch.save(checkpoint, "checkpoint/checkpoint-epoch{:04d}.pth".format(epoch))


def train_G(G, D, G_optimizer):
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z)

    x_fake_d_logit = D(x_fake)
    G_loss = g_loss_fn(x_fake_d_logit)

    G.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    return {'g_loss': G_loss}


def train_D(x_real, G, D):
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z).detach()

    x_real_d_logit = D(x_real)
    x_fake_d_logit = D(x_fake)

    x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
    gp = gan.gradient_penalty(functools.partial(D), x_real, x_fake, gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)

    D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, summary_writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    batch_size = data_loader.batch_sampler.batch_size
    n_samples = len(data_loader)

    for step, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, list)} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # if lr_scheduler is not None:
        #     lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metric_logger.to_summary_writer(
            summary_writer, 'training_step', epoch*(n_samples//batch_size)+step)

    metric_logger.to_summary_writer(summary_writer, 'training_epoch', epoch)
    return metric_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('-c', '--checkpoints_dir', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--batch_size', type=int, default=8, help='default: {:g}'.format(8))
    parser.add_argument('--learning_rate', type=float, default=0.003, help='default: {:g}'.format(0.003))
    parser.add_argument('--momentum', type=float, default=0.9, help='default: {:g}'.format(0.9))
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='default: {:g}'.format(0.0005))
    parser.add_argument('--num_steps_to_display', type=int, default=20, help='default: {:d}'.format(20))
    parser.add_argument('--num_epochs_to_snapshot', type=int, default=1, help='default: {:d}'.format(1))
    parser.add_argument('--num_checkpoints_to_reserve', type=int, default=10, help='default: {:d}'.format(10))
    parser.add_argument('--epochs', type=int, default=100, help='default: {:d}'.format(100))
    parser.add_argument('--current_epoch', type=int, default=1, help='default: {:d}'.format(1))
    parser.add_argument('--workers', type=int, default=4, help='default: {:d}'.format(4))
    parser.add_argument('--intra_class', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='coco2017', help='default: {:s}'.format('coco2017'))
    args = parser.parse_args()

    args.num_categories = 3
    args.z_dim = 1
    args.x_dim = 1
    args.n_D_downsamplings = 6
    args.adversarial_loss_mode = 'gan'
    args.gradient_penalty_mode = 'none'
    args.gradient_penalty_sample_mode = 'line'
    args.gradient_penalty_weight = 10.0
    trainer = Trainer(args)

    for epoch in range(args.current_epoch, args.epochs+1):
        # train for one epoch, printing every 10 iterations
        trainer.train_one_epoch(epoch)

        # update the learning rate
        trainer.lr_scheduler.step()

        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

        trainer.snapshot(epoch)

    print("That's it!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
#     parser.add_argument('-d', '--dataset_dir', type=str, default='./data', help='path to data directory')
#     parser.add_argument('-c', '--checkpoints_dir', type=str, default='./checkpoint', help='path to outputs directory')
#     parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
#     parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
#     parser.add_argument('--batch_size', type=int, default=8, help='default: {:g}'.format(8))
#     parser.add_argument('--learning_rate', type=float, default=0.003, help='default: {:g}'.format(0.003))
#     parser.add_argument('--momentum', type=float, default=0.9, help='default: {:g}'.format(0.9))
#     parser.add_argument('--weight_decay', type=float, default=0.0005, help='default: {:g}'.format(0.0005))
#     parser.add_argument('--num_steps_to_display', type=int, default=20, help='default: {:d}'.format(20))
#     parser.add_argument('--num_epochs_to_snapshot', type=int, default=1, help='default: {:d}'.format(1))
#     parser.add_argument('--num_checkpoints_to_reserve', type=int, default=10, help='default: {:d}'.format(10))
#     parser.add_argument('--epochs', type=int, default=100, help='default: {:d}'.format(100))
#     parser.add_argument('--current_epoch', type=int, default=1, help='default: {:d}'.format(1))
#     parser.add_argument('--workers', type=int, default=4, help='default: {:d}'.format(4))
#     parser.add_argument('--intra_class', action='store_true')
#     parser.add_argument('--dataset_name', type=str, default='coco2017', help='default: {:s}'.format('coco2017'))
#     args = parser.parse_args()

#     # train on the GPU or on the CPU, if a GPU is not available
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     ct_transform = T.Compose([T.ToTensor()])

#     dataset = LiTSDataset(args.dataset_dir, ct_transform)
#     # define sampler
#     sampler = torch.utils.data.RandomSampler(dataset)
#     batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)

#     # define training and validation data loaders
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_sampler=batch_sampler, num_workers=args.workers,
#         collate_fn=collate_fn)

#     # data_loader_test = torch.utils.data.DataLoader(
#     #     dataset_test, batch_size=2, shuffle=False,  # num_workers=4,
#     #     collate_fn=utils.collate_fn)

#     # get the model using our helper function
#     backbone_name = os.path.basename(args.backbone_path).split('-')[0]
#     # number of categories includes background
#     num_categories = 3
#     # model = MaskRCNN(backbone, num_categories, min_size=args.image_min_side, max_size=args.image_max_side)# TODO: why wrong?
#     model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_categories)
#     # move model to the right device
#     model.to(device)

#     # construct an optimizer
#     params = [p for p in model.parameters() if p.requires_grad]

#     # SGD
#     optimizer = torch.optim.SGD(params, lr=args.learning_rate,
#                                 momentum=args.momentum, weight_decay=args.weight_decay)

#     # and a learning rate scheduler
#     lr_scheduler = WarmUpMultiStepLR(optimizer, milestones=[60, 80], gamma=0.1,
#                                      factor=0.3333, num_iters=5)
#     # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
#     summary_writer = torch.utils.tensorboard.SummaryWriter('logs')

#     checkpoint_path = os.path.join(args.checkpoints_dir, 'checkpoint-epoch{:04d}.pth'.format(args.current_epoch-1))
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_=device)
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         model.load_state_dict(checkpoint['state_dict'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         print('checkpoint loaded.')
#     else:
#         print('{:s} does not exist.'.format(checkpoint_path))

#     for epoch in range(args.current_epoch, args.epochs+1):
#         model.train()
#         # train for one epoch, printing every 10 iterations
#         train_one_epoch(
#             model, optimizer, data_loader, device, epoch,
#             print_freq=args.num_steps_to_display,
#             summary_writer=summary_writer
#         )

#         # update the learning rate
#         lr_scheduler.step()

#         # evaluate on the test dataset
#         # evaluate(model, data_loader_test, device=device)

#         if epoch % args.num_epochs_to_snapshot == 0:
#             os.makedirs('checkpoint', exist_ok=True)
#             ckpts_path = glob.glob('checkpoint/checkpoint-epoch*.pth')
#             if len(ckpts_path) > args.num_checkpoints_to_reserve:
#                 for path in ckpts_path[:len(ckpts_path)-args.num_checkpoints_to_reserve]:
#                     os.remove(path)
#             checkpoint = {
#                 'epoch': epoch,
#                 'optimizer': optimizer.state_dict(),
#                 'state_dict': model.state_dict(),
#                 'lr_scheduler': lr_scheduler.state_dict()
#             }
#             torch.save(checkpoint, "checkpoint/checkpoint-epoch{:04d}.pth".format(epoch))

#     print("That's it!")
