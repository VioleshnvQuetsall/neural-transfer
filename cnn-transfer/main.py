import time
import datetime
import os
from pprint import pprint

os.chdir(os.path.dirname(__file__))

import numpy as np

from PIL import Image

import torch
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from models import *
from utils import *


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

opt = parse_options('options.yaml')
pprint(opt[None])

timestamp = opt.project.timestamp
timestamp = timestamp and datetime.datetime.now().strftime(timestamp)
img_dir, log_dir, output_dir = init_dir(opt.project.assets,
                                        opt.project.name,
                                        timestamp)
print(f'img_dir: {img_dir}, log_dir: {log_dir}, output_dir: {output_dir}')


writer = SummaryWriter(log_dir=log_dir)
if opt.project.clear_log:
    clear_log(log_dir)
if opt.epoch.start == 0 and opt.project.clear_output:
    clear_log(output_dir)


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

unnormalize = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_images():
    # ------
    #  Load
    # ------
    content_path, style_path = None, None
    for file in (f for f in os.listdir(img_dir)
                 if os.path.isfile(os.path.join(img_dir, f))):
        if file.startswith('content'):
            content_path = os.path.join(img_dir, file)
        elif file.startswith('style'):
            style_path = os.path.join(img_dir, file)

    if content_path is None or style_path is None:
        raise FileNotFoundError('content image or style image not found')

    # -----------
    #  Transform
    # -----------
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')

    print(f'load content: {content_path}, style: {style_path}')

    content_img = transforms.Resize(opt.image.content_imsize)(content_img)
    content_img = transform(content_img).unsqueeze(0).type(Tensor)
    style_img = transforms.Resize(opt.image.style_imsize)(style_img)
    style_img = transform(style_img).unsqueeze(0).type(Tensor)

    if opt.epoch.start != 0:
        generate_img = transform(Image.open(
            os.path.join(output_dir, f'{opt.epoch.start}.jpg')))
        generate_img = generate_img.unsqueeze(0)
        print(f'load {output_dir}/{opt.epoch.start}.jpg')
    else:
        if opt.image.init == 'content':
            generate_img = content_img.clone()
        elif opt.image.init == 'noise':
            generate_img = torch.randn(content_img.shape)
        else:
            raise ValudError(f'invalid init method: {opt.image.init}')

    generate_img = generate_img.type(Tensor)
    if cuda:
        content_img = content_img.cuda()
        style_img = style_img.cuda()
        generate_img = generate_img.cuda()

    generate_img.requires_grad_(True)

    return content_img, style_img, generate_img

content_img, style_img, generate_img = get_images()
print(f'C: {content_img.shape} '
      f'S: {style_img.shape} '
      f'G: {generate_img.shape} ')

model, content_losses, style_losses = get_model(content_img,
                                                style_img,
                                                opt.model.content_layers,
                                                opt.model.style_layers,
                                                opt.model.content_weights,
                                                opt.model.style_layers)


def get_optimizer_scheduler():
    optimizer = torch.optim.LBFGS([generate_img],
                                  lr=opt.optimizer.lr,
                                  max_iter=opt.optimizer.max_iter)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       opt.scheduler.gamma)

    return optimizer, scheduler

optimizer, scheduler = get_optimizer_scheduler()


def save_sample(img, name):
    with torch.no_grad():
        img = unnormalize(img.detach()).squeeze()
    save_image(img, os.path.join(output_dir, name), normalize=False)
    print(f'save image: {name}')
    return img


def train_and_sample():
    if opt.epoch.start == 0:
        writer.add_image('generate', save_sample(generate_img, '-1.jpg'), -1)
    prev_time = time.time()
    for epoch in range(opt.epoch.start, opt.epoch.n + 1):
        display_content_loss = []
        display_style_loss = []
        def closure():
            # ----------------
            #  Calculate Loss
            # ----------------

            optimizer.zero_grad()
            model(generate_img)

            content_loss = sum(lc.loss for lc in content_losses)
            style_loss = sum(ls.loss for ls in style_losses)

            loss = opt.loss.alpha * content_loss + opt.loss.beta * style_loss
            loss.backward()

            # --------------
            #  Log Progress
            # --------------

            print(f'[Epoch: {epoch}/{opt.epoch.n}] '
                  f'[total loss: {loss.item():.4f}] '
                  f'[content loss: {content_loss.item():.4f}] '
                  f'[style loss: {style_loss.item():.4f}] ')
            display_content_loss.append(content_loss.item())
            display_style_loss.append(style_loss.item())

            return loss

        optimizer.step(closure)
        scheduler.step()

        # ---------------------
        #  TensorBoard Display
        # ---------------------

        batches_left = opt.epoch.n - epoch
        curr_time = time.time()
        time_left = datetime.timedelta(seconds=batches_left * (curr_time - prev_time))
        prev_time = curr_time
        print(f'[ETA: {time_left}]')

        writer.add_scalar('Loss/Content', np.mean(display_content_loss), epoch)
        writer.add_scalar('Loss/Style', np.mean(display_style_loss), epoch)
        if epoch != 0 and epoch % opt.epoch.sample_interval == 0:
            writer.add_image('generate',
                             save_sample(generate_img, f'{epoch}.jpg'),
                             epoch)
        writer.flush()


if __name__ == '__main__':
    train_and_sample()
