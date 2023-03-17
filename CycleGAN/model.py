import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Conv Block for Generator
    """
    def __init__(self, in_channels, out_channels, downsample=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if downsample
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    """
    Residual Block for Generator, as bottleneck layers
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorCycleGAN(nn.Module):
    """
    Generator for CycleGAN
    """
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2, num_features * 4, kernel_size=3, stride=2,padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4, num_features * 2, downsample=False, kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                ConvBlock(
                    num_features * 2, num_features * 1, downsample=False, kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

class Block(nn.Module):
    """
    Discriminator block
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect",),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DiscriminatorCycleGAN(nn.Module):
    """
    Discriminator for CycleGAN
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect",),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        # append blocks with wider channels
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        # the final layer is a convolution with 1 channel
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial(x)
        return self.sigmoid(self.model(x))
    
def test1_generator():
    img_channels = 3
    img_size = 128
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels)
    assert gen(x).shape == x.shape, "Generator output shape is not correct"
    print("Generator test1 passed!")

def test2_discriminator():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    assert preds.shape == (5, 1, 30, 30), "Discriminator output shape is not correct"
    print("Discriminator test2 passed!")

if __name__ == "__main__":
    test1_generator()
    test2_discriminator()