from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR


class WarmUpMultiStepLR(MultiStepLR):
    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1,
                 factor: float = 0.3333, num_iters: int = 500, last_epoch: int = -1):
        self.factor = factor
        self.num_iters = num_iters
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        super(WarmUpMultiStepLR, self).__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.num_iters:
            alpha = self.last_epoch / self.num_iters
            factor = (1 - self.factor) * alpha + self.factor
            return [lr * factor for lr in self.base_lr]
        else:
            return super(WarmUpMultiStepLR, self).get_lr()
