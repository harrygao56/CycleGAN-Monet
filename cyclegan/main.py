import numpy as np
import itertools
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
from PIL import Image
from utils import *
from cyclegan import *
from train import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

params = {
    "n_epochs": 200,
    "batch_size": 4,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "img_size": 256,
    "channels": 3,
    "num_residual_blocks": 19,
    "lambda_cycle": 10.0,
    "lambda_id": 5.0
}

root = "data"
transforms_ = [
    transforms.Resize((params['img_size'], params['img_size']), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader = DataLoader(
    ImageDataset(root, mode="train", transforms_=transforms_),
    batch_size=params['batch_size'],
    shuffle=True,
    num_workers=1,
)
val_dataloader = DataLoader(
    ImageDataset(root, mode="test", transforms_=transforms_),
    batch_size=16,
    shuffle=True,
    num_workers=1,
)

gan_loss = torch.nn.MSELoss()
cycle_loss = torch.nn.L1Loss()
id_loss = torch.nn.L1Loss()
input_shape = (params['channels'], params['img_size'], params['img_size'])

gen_G = GeneratorResNet(input_shape, params['num_residual_blocks'])
gen_F = GeneratorResNet(input_shape, params['num_residual_blocks'])
dX = Discriminator(input_shape) 
dY = Discriminator(input_shape)

gen_G = gen_G.to(device)
gen_F = gen_F.to(device)
dX = dX.to(device)
dY = dY.to(device)

gen_G.apply(initialize_conv_weights_normal)
gen_F.apply(initialize_conv_weights_normal)
dX.apply(initialize_conv_weights_normal)
dY.apply(initialize_conv_weights_normal)

buffer_X = ReplayBuffer()
buffer_Y = ReplayBuffer()

optimizer_G = torch.optim.Adam(
    itertools.chain(gen_G.parameters(), gen_F.parameters()),
    lr=params['lr'],
    betas=(params['b1'], params['b2']),
)
optimizer_dX = torch.optim.Adam(dX.parameters(), lr=params['lr'], betas=(params['b1'], params['b2']))
optimizer_dY = torch.optim.Adam(dY.parameters(), lr=params['lr'], betas=(params['b1'], params['b2']))

train(gen_G, gen_F, dX, dY, train_dataloader, val_dataloader, params['n_epochs'], id_loss, cycle_loss, gan_loss, params['lambda_cycle'], params['lambda_id'], optimizer_G, optimizer_dX, optimizer_dY, buffer_X, buffer_Y, device)