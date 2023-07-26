import os
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import glob
import random
from torch.utils.data import Dataset
from PIL import Image


# Methods for Image Dataloader


def convert_to_RGB(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        if mode == 'train':
            self.monet_files = sorted(glob.glob(os.path.join(root, "monet_jpg") + "/*.*")[:600])
            self.photo_files = sorted(glob.glob(os.path.join(root, "photo_jpg") + "/*.*")[:250])
        elif mode == 'test':
            self.monet_files = sorted(glob.glob(os.path.join(root, "monet_jpg") + "/*.*")[250:])
            self.photo_files = sorted(glob.glob(os.path.join(root, "photo_jpg") + "/*.*")[250:300])
        elif mode == 'all':
            self.monet_files = sorted(glob.glob(os.path.join(root, "monet_jpg") + "/*.*"))
            self.photo_files = sorted(glob.glob(os.path.join(root, "photo_jpg") + "/*.*"))

    def __getitem__(self, index):
        monet = Image.open(self.monet_files[index % len(self.monet_files)])
        photo = Image.open(self.photo_files[random.randint(0, len(self.photo_files) - 1)])

        if monet.mode != "RGB":
            monet = convert_to_RGB(monet)
        if photo.mode != "RGB":
            photo = convert_to_RGB(photo)

        monet = self.transform(monet)
        photo = self.transform(photo)

        return (monet.float(), photo.float())

    def __len__(self):
        return max(len(self.monet_files), len(self.photo_files))


# Implement replay buffer, which is implemented by the paper to improve the training stability of the discriminator
class ReplayBuffer:
    # Create image buffer to store previous 50 images
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def initialize_conv_weights_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)