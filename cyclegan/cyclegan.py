import torch.nn as nn

# CycleGAN archicture implemented from Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
# https://arxiv.org/abs/1703.10593


# Residual block with two convolution layers
class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
        )

    def forward(self, x):
        return x + self.block(x)


# Generator from CycleGAN paper
# c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,R256,R256,R256,u128 u64,c7s1-3
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_channels = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_channels, kernel_size = 7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        in_channels = out_channels

        # Downsampling (Encoder)
        for _ in range(2):
            out_channels *= 2
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_channels)]

        # Upsampling (Decoder)
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_channels, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Discriminator from CycleGAN paper
# C64-C128-C256-C512
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of discriminator
        self.output_shape = (1, height //2 ** 4, width // 2 ** 4)

        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, out_channels=64, normalize=False),
            *discriminator_block(64, out_channels=128),
            *discriminator_block(128, out_channels=256),
            *discriminator_block(256, out_channels=512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
        )

    def forward(self, img):
        return self.model(img)