from functools import partial

import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

cuda = torch.cuda.is_available()


class UnNormalize:
    def __init__(self, mean, std):
        mean = torch.tensor(mean).view((1, -1, 1, 1))
        std = torch.tensor(std).view((1, -1, 1, 1))
        if cuda:
            mean = mean.cuda()
            std = std.cuda()

        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = (x * self.std) + self.mean
        x.clamp_(0, 1)
        return x


def gram(feature_maps, scale=False):
    n, c, h, w = feature_maps.shape
    feature = feature_maps.view(n * c, h * w)
    gram_matrix = torch.mm(feature, feature.t())
    if scale:
        gram_matrix /= n * c * h * w
    return gram_matrix


def scale_mse_loss(x, target):
    diff = (x - target)
    return diff.pow(2).sum() / diff.abs().sum().add(1e-8)


class ContentLoss(nn.Module):

    def __init__(self, target, weight=1):
        super().__init__()
        self.target = target.detach()

        self.weight = weight
        # self.loss_fn = scale_mse_loss
        self.loss_fn = partial(F.mse_loss, reduction='mean')
        self.loss = None

        self.activate = False

    def forward(self, x):
        if self.activate:
            self.loss = self.weight * self.loss_fn(x, self.target)
        return x


class StyleLoss(nn.Module):

    def __init__(self, target, weight=1):
        super().__init__()
        self.target = target
        self.gram_target = gram(target, True).detach()

        self.weight = weight
        # self.loss_fn = scale_mse_loss
        self.loss_fn = partial(F.mse_loss, reduction='sum')
        self.loss = None

        self.activate = False

    def forward(self, x):
        if self.activate:
            self.loss = self.weight * self.loss_fn(gram(x, True), self.gram_target)
        return x


def get_model(content_img, style_img,
              content_layers, style_layers,
              content_weights=None, style_weights=None):
    # assert style_img.shape == content_img.shape

    content_weights = content_weights or [1] * len(content_layers)
    style_weights = style_weights or [1] * len(style_layers)
    content_weights = [cw / sum(content_weights) for cw in content_weights]
    style_weights = [sw / sum(style_weights) for sw in style_weights]

    # -------------
    #  Build Model
    # -------------
    conv_count = max(max(content_layers), max(style_layers))
    if conv_count > 16:
        raise Warning(f'content layers or style layers exceed conv layer count (16)')

    if cuda:
        vgg = models.vgg19(pretrained=True).cuda().eval()
    else:
        vgg = models.vgg19(pretrained=True).eval()
    cnn = vgg.features

    model = nn.Sequential()
    content_losses = []
    style_losses = []

    i = 0
    conv_before = False
    for j, layer in enumerate(cnn.children()):
        model.add_module(f'vgg {j}', layer)

        if isinstance(layer, nn.ReLU) and conv_before:
            i += 1
            if i in content_layers:
                target = model(content_img).detach()
                lc = ContentLoss(target, content_weights[content_layers.index(i)])
                model.add_module(f'content {i}', lc)
                content_losses.append(lc)
            if i in style_layers:
                target = model(style_img).detach()
                ls = StyleLoss(target, style_weights[style_layers.index(i)])
                model.add_module(f'style {i}', ls)
                style_losses.append(ls)
            if i >= conv_count:
                break
        conv_before = isinstance(layer, nn.Conv2d)

    for loss in content_losses + style_losses:
        loss.activate = True

    if cuda:
        model = model.cuda()
    model.requires_grad_(False)

    return model, content_losses, style_losses


