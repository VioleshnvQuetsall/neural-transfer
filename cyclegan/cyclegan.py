import time
import datetime
import os

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from cycle_models import *
from dataset import *
from utils import *


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

opt = parse_options('configs/cycle_options.yaml')
logger = get_logger('configs/cycle_logger.yaml')

transform = default_transform(opt.img_height, opt.img_width)

input_shape = tuple(map(opt.__getattr__, ['channels', 'img_height', 'img_width']))
logger.debug(f'input_shape: {input_shape}')


def get_GD():
    from itertools import chain
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
    if opt.epoch != 0:
        G_AB.load_state_dict(torch.load(f'models/{opt.name}/G_AB{opt.epoch}'))
        G_BA.load_state_dict(torch.load(f'models/{opt.name}/G_BA{opt.epoch}'))
        D_A.load_state_dict(torch.load(f'models/{opt.name}/D_A{opt.epoch}'))
        D_B.load_state_dict(torch.load(f'models/{opt.name}/D_B{opt.epoch}'))
        logger.info(f'load {opt.epoch}')
    else:
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
    
    optimizer_G = torch.optim.Adam(chain(G_AB.parameters(), G_BA.parameters()),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(),
                                     lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(),
                                     lr=opt.lr, betas=(opt.b1, opt.b2))
    
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    return ((G_AB, G_BA),
            (D_A, D_B),
            (optimizer_G, optimizer_D_A, optimizer_D_B),
            (lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B))

def get_criterions():
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    if cuda:
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
    return criterion_GAN, criterion_cycle, criterion_identity

def get_dataloader():
    root = opt.root
    if not os.path.isdir(root):
        raise FileNotFoundError(f'Not such dictionary: {root}')
    # [0.5199, 0.5115, 0.4722] [0.2257, 0.2176, 0.2435]
    train_dataloader = DataLoader(ImagePairDataset(root, transform, unaligned=True, mode='train'),
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.n_cpu)
    # [0.4123, 0.4093, 0.3927] [0.2711, 0.2487, 0.2778]
    test_dataloader = DataLoader(ImagePairDataset(root, transform, unaligned=True, mode='test'),
                                 batch_size=5,
                                 shuffle=True,
                                 num_workers=1)
    return train_dataloader, test_dataloader


G, D, optimizers, lr_schedulers = get_GD()
G_AB, G_BA = G
D_A, D_B = D
optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B = lr_schedulers

criterion_GAN, criterion_cycle, criterion_identity = get_criterions()
dataloader, val_dataloader = get_dataloader()

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


def sample_images(batch_count):
    with torch.no_grad():
        imgs = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = Variable(imgs['A'].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs['B'].type(Tensor))
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=False) * 0.5 + 0.5
        real_B = make_grid(real_B, nrow=5, normalize=False) * 0.5 + 0.5
        fake_A = make_grid(fake_A, nrow=5, normalize=False) * 0.5 + 0.5
        fake_B = make_grid(fake_B, nrow=5, normalize=False) * 0.5 + 0.5
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

        path = f'output/{opt.name}'
        if not os.path.isdir(path):
            os.mkdir(path)
        name = os.path.join(path, f'{batch_count}.png')
        save_image(image_grid, name, normalize=False)
        logger.info(f'save sample: {name}')


def train_and_sample():
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch['A'].type(Tensor))
            real_B = Variable(batch['B'].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))),
                             requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))),
                            requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            for _ in range(opt.n_train_D):

                # -----------------------
                #  Train Discriminator A
                # -----------------------

                optimizer_D_A.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2

                loss_D_A.backward()
                optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------

                optimizer_D_B.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2

                loss_D_B.backward()
                optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Log
            (logger.info if i == 0 else logger.debug)(
                f'[Epoch: {epoch}/{opt.n_epochs}] '
                f'[Batch: {i}/{len(dataloader)}] '
                f'[loss_D: {loss_D.item():.4f}] '
                f'[loss_G: {loss_G.item():.4f}, adv: {loss_GAN.item():.4f},'
                f' cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}]'
                f'[ETA: {time_left}]')

            # If at sample interval save image
            if batches_done != 0 and batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch != 0 and epoch % opt.checkpoint_interval == 0:
            path = f'models/{opt.name}'
            # Save model checkpoints
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(G_AB.state_dict(), os.path.join(path, f'G_AB{epoch}'))
            torch.save(G_BA.state_dict(), os.path.join(path, f'G_BA{epoch}'))
            torch.save(D_A.state_dict(), os.path.join(path, f'D_A{epoch}'))
            torch.save(D_B.state_dict(), os.path.join(path, f'D_B{epoch}'))
            logger.info(f'save {epoch}')

if __name__ == '__main__':
    train_and_sample()
