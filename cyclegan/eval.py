import numpy as np
import torch
import matplotlib.pyplot as plt
from cyclegan import *
from utils import *
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_residual_layers = 19
input_shape = (3, 256, 256)
root = "data"
download_path = "outputs"
transforms_ = [
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(root, transforms_, "all"),
    batch_size=4,
    shuffle=True,
    num_workers=1
)

generator = GeneratorResNet(input_shape, num_residual_layers)
state_dict = torch.load("generator_G_final", map_location=device)
generator.load_state_dict(state_dict)
generator.eval()

for i, (monet, photo) in enumerate(dataloader):
    outputs = generator(photo)
    outputs = np.transpose(outputs.cpu().detach().numpy(), [0, 2, 3, 1])
    outputs = outputs / 2 + 0.5
    for j in range(outputs.shape[0]):
        output = (outputs[j, :, :, :] * 255).astype(np.uint8)
        im = Image.fromarray(output).convert('RGB')
        im.show()
        # im.save(f'{download_path}/output_img_{i}_{j}.jpg')
