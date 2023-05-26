from functools import partial

import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

cuda = torch.cuda.is_available()


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        mean = torch.tensor(mean).view((1, -1, 1, 1))
        std = torch.tensor(std).view((1, -1, 1, 1))
        if cuda:
            mean = mean.cuda()
            std = std.cuda()

        self.mean = mean
        self.std = std

    def forward(self, x):
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

    def __init__(self, weight=1):
        super().__init__()
        self.target = None

        self.weight = weight
        # self.loss_fn = scale_mse_loss
        self.loss_fn = partial(F.mse_loss, reduction='mean')
        self.loss = None

        # activate after loading content
        self.load_content = False
        self.activate = False

    def forward(self, x):
        if self.activate:
            self.loss = self.weight * self.loss_fn(x, self.target)
        elif self.load_content:
            self.target = x.detach()
        return x


class StyleLoss(nn.Module):

    def __init__(self, weight=1):
        super().__init__()

        self.target, self.gram_target = None, None

        self.weight = weight
        # self.loss_fn = scale_mse_loss
        self.loss_fn = partial(F.mse_loss, reduction='sum')
        self.loss = None

        # activate after loading style
        self.load_style = False
        self.activate = False

    def forward(self, x):
        if self.activate:
            self.loss = self.weight * self.loss_fn(gram(x, True), self.gram_target)
        elif self.load_style:
            self.gram_target = gram(x, True).detach()
        return x


def get_model(content_img, style_img,
              content_layers, style_layers,
              content_weights=None, style_weights=None):

    # process loss weights
    content_weights = content_weights or [1] * len(content_layers)
    style_weights = style_weights or [1] * len(style_layers)
    content_weights = [cw / sum(content_weights) for cw in content_weights]
    style_weights = [sw / sum(style_weights) for sw in style_weights]

    # -------------
    #  Build Model
    # -------------
    conv_count = max(max(content_layers), max(style_layers))
    if conv_count > 16:
        raise Warning(f'content layers or style layers '
                      f'exceed the number of conv layer '
                      f'(conv_count {conv_count} > 16)')

    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    vgg = vgg.cuda().eval() if cuda else vgg.eval()
    cnn = vgg.features

    model = nn.Sequential()
    content_losses = []      # contains ContentLoss in model
    style_losses = []        # contains StyleLoss in model

    # build network according to content_layers and style_layers
    i = 0
    conv_before = False
    for j, layer in enumerate(cnn.children()):
        model.add_module(f'vgg {j}', layer)

        if isinstance(layer, nn.ReLU) and conv_before:
            # add loss layer after conv_relu layer
            i += 1
            if i in content_layers:
                lc = ContentLoss(content_weights[content_layers.index(i)])
                model.add_module(f'content {i}', lc)
                content_losses.append(lc)
            if i in style_layers:
                ls = StyleLoss(style_weights[style_layers.index(i)])
                model.add_module(f'style {i}', ls)
                style_losses.append(ls)
            if i >= conv_count:
                # no more loss layer
                break
        conv_before = isinstance(layer, nn.Conv2d)

    # initial content losses
    for loss in content_losses:
        loss.load_content = True
    model(content_img)
    for loss in content_losses:
        loss.load_content = False

    # initial style losses
    for loss in style_losses:
        loss.load_style = True
    model(style_img)
    for loss in style_losses:
        loss.load_style = False

    # if actiavte before model building,
    # error may be raised due to the mismatched size of content and style,
    # which results from the attempt to calculate loss between content and style
    for loss in content_losses + style_losses:
        loss.activate = True

    if cuda:
        model = model.cuda()

    # we want to optimize generated image, instead of model
    model.requires_grad_(False)

    # pass image to model, and get losses from two lists of loss layers
    return model, content_losses, style_losses
