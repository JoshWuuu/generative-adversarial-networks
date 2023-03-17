"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from model import *

def train_fn(
    gen_Z, disc_Z, gen_H, disc_H, loader, opt_disc, opt_gen, l1, mse, 
    d_scaler, g_scaler
):
    """
    training function for CycleGAN

    Inputs:
    - gen_Z: model, generator for zebra images
    - disc_Z: model, discriminator for zebra images
    - gen_H: model, generator for horse images
    - disc_H: model, discriminator for horse images
    - loader: torch.utils.data.DataLoader, dataloader for training data
    - opt_disc: torch.optim, optimizer for discriminator
    - opt_gen: torch.optim, optimizer for generator
    - l1: torch.nn.modules.loss.L1Loss, loss function for cycle consistency
    - mse: torch.nn.modules.loss.MSELoss, loss function for adversarial loss    
    - d_scaler: torch.cuda.amp.GradScaler, scaler for discriminator
    - g_scaler: torch.cuda.amp.GradScaler, scaler for generator

    """
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            # discriminator for fake zebra
            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            # the adversarial loss for the discriminator is using mse, not bce
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # discriminator for fake horse
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        # update discriminator using optimizer and scaler
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            #  l1 loss between the original image and the image that is generated from the generated image
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss 
            # l1 loss between the original image and the image that is generated from the original image
            # loss for updating the generator to not change the irrelevent pixels 
            # remove these for efficiency if you set lambda_identity=0
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z + loss_G_H
                + config.LAMBDA_CYCLE * (cycle_zebra_loss + cycle_horse_loss)
                + config.LAMBDA_IDENTITY * (identity_horse_loss + identity_zebra_loss)
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(D_loss = D_loss, G_loss=G_loss)