import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
from train import *
import config

def main():
    # If you train on MNIST, remember to set channels_img to 1
    dataset = datasets.MNIST(
        root="dataset/", train=True, transform=config.transforms, download=True
    )

    # comment mnist above and uncomment below if train on CelebA
    # dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    gen = GeneratorDCGAN(config.NOISE_DIM, config.CHANNELS_IMG, config.FEATURES_GEN).to(config.device)
    disc = DiscriminatorDCGAN(config.CHANNELS_IMG, config.FEATURES_DISC).to(config.device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    train_fn(
        gen, disc, dataloader, criterion, opt_gen, opt_disc, 
        config.device, config.NUM_EPOCHS, config.BATCH_SIZE, config.NOISE_DIM
    )

if __name__ == "__main__":
    main()