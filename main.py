import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
from train import *

def main():
    # Hyperparameters etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 1
    NOISE_DIM = 100
    NUM_EPOCHS = 5
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

    # If you train on MNIST, remember to set channels_img to 1
    dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms, download=True
    )

    # comment mnist above and uncomment below if train on CelebA
    # dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gen = GeneratorDCGAN(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = DiscriminatorDCGAN(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    train_fn(gen, disc, dataloader, criterion, opt_gen, opt_disc, device, NUM_EPOCHS, BATCH_SIZE, NOISE_DIM)

if __name__ == "__main__":
    main()