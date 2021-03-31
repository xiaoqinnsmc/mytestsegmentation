import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Generator(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(Generator, self).__init__()

        n1 = 4
        depth = 9
        filters = [n1 * 2 ** i for i in range(depth)]

        self.mask_first_conv = conv_block(in_ch, filters[0])
        self.image_first_conv = conv_block(in_ch, filters[0])

        # down scaling
        conv_blocks = [conv_block(filters[i], filters[i + 1]) for i in range(depth - 1)]
        self.mask_conv_blocks = nn.ModuleList(conv_blocks)

        conv_blocks = [conv_block(filters[i], filters[i + 1]) for i in range(depth - 1)]
        self.image_conv_blocks = nn.ModuleList(conv_blocks)

        # fc in
        self.fc1 = nn.Linear(in_features=4096, out_features=512)
        # fc out
        self.fc2 = nn.Linear(in_features=1536, out_features=4096)

        # up scaling
        up_convs = [up_conv(filters[depth - i], filters[depth - i - 1]) for i in range(1, depth)]
        self.up_convs = nn.ModuleList(up_convs)
        conv_blocks = [conv_block(filters[depth - i], filters[depth - i - 1]) for i in range(1, depth)]
        self.up_conv_blocks = nn.ModuleList(conv_blocks)

        self.tail = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x, y):
        y = self.mask_first_conv(y)
        x = self.image_first_conv(x)
        image_features, mask_features = [x], [y]

        for conv_block in self.mask_conv_blocks:
            y = F.max_pool2d(conv_block(y), kernel_size=2, stride=2)
            mask_features.append(y)

        for conv_block in self.image_conv_blocks:
            x = F.max_pool2d(conv_block(x), kernel_size=2, stride=2)
            image_features.append(x)

        y, x = torch.flatten(mask_features[-1], start_dim=1), torch.flatten(image_features[-1], start_dim=1)
        y, x = self.fc1(y), self.fc1(x)
        latent = torch.randn_like(y)
        y = torch.cat((x, y, latent), dim=1)  # TODO fc bn
        y = F.leaky_relu(self.fc2(y)).view_as(mask_features[-1])

        for i, (up, up_b) in enumerate(zip(self.up_convs, self.up_conv_blocks), start=2):
            y = up(y)
            y = torch.cat((image_features[-i], y), dim=1)
            y = up_b(y)

        out = F.sigmoid(self.tail(y))

        return out


class Discriminator(nn.Module):

    def __init__(self,
                 input_channels=3,
                 dim=64,
                 n_downsamplings=4):
        super().__init__()

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            )

        layers = []

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))

        # 2: logit
        self.tail = nn.Conv2d(d, d, kernel_size=4, stride=1, padding=0)
        self.convs = nn.ModuleList(layers)

    def forward(self, x):
        for module in self.convs:
            x = module(x)
        y = self.tail(x)
        return y


def get_gan_losses_fn():
    bce = torch.nn.BCEWithLogitsLoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(r_logit, torch.ones_like(r_logit))
        f_loss = bce(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = torch.max(1 - f_logit, torch.zeros_like(f_logit)).mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = torch.nn.MSELoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(r_logit, torch.ones_like(r_logit))
        f_loss = mse(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = -r_logit.mean()
        f_loss = f_logit.mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'hinge_v2':
        return get_hinge_v2_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()


# ======================================
# =           sample method            =
# ======================================

def _sample_line(real, fake):
    shape = [real.size(0)] + [1] * (real.dim() - 1)
    alpha = torch.rand(shape, device=real.device)
    sample = real + alpha * (fake - real)
    return sample


def _sample_DRAGAN(real, fake):  # fake is useless
    beta = torch.rand_like(real)
    fake = real + 0.5 * real.std() * beta
    sample = _sample_line(real, fake)
    return sample


# ======================================
# =      gradient penalty method       =
# ======================================

def _norm(x):
    norm = x.view(x.size(0), -1).norm(p=2, dim=1)
    return norm


def _one_mean_gp(grad):
    norm = _norm(grad)
    gp = ((norm - 1)**2).mean()
    return gp


def _zero_mean_gp(grad):
    norm = _norm(grad)
    gp = (norm**2).mean()
    return gp


def _lipschitz_penalty(grad):
    norm = _norm(grad)
    gp = (torch.max(torch.zeros_like(norm), norm - 1)**2).mean()
    return gp


def gradient_penalty(f, real, fake, gp_mode, sample_mode):
    sample_fns = {
        'line': _sample_line,
        'real': lambda real, fake: real,
        'fake': lambda real, fake: fake,
        'dragan': _sample_DRAGAN,
    }

    gp_fns = {
        '1-gp': _one_mean_gp,
        '0-gp': _zero_mean_gp,
        'lp': _lipschitz_penalty,
    }

    if gp_mode == 'none':
        gp = torch.tensor(0, dtype=real.dtype, device=real.device)
    else:
        x = sample_fns[sample_mode](real, fake).detach()
        x.requires_grad = True
        pred = f(x)
        grad = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
        gp = gp_fns[gp_mode](grad)

    return gp
