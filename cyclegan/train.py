import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *
from cyclegan import *
from PIL import Image


def train(gen_G, gen_F, dX, dY, dataloader, val_dataloader, n_epochs, id_loss, cycle_loss, gan_loss, 
          lambda_cycle, lambda_id, optimizer_G, optimizer_dX, optimizer_dY, buffer_X, buffer_Y, device):
    for epoch in range(n_epochs):
        for i, (monet, photo) in enumerate(dataloader):
            monet = monet.to(device)
            photo = photo.to(device)

            valid = torch.from_numpy(np.ones((monet.size(0), *dX.output_shape), dtype="float32")).to(device)
            generated = torch.from_numpy(np.zeros((monet.size(0), *dX.output_shape), dtype="float32")).to(device)

            # TRAIN GENERATORS
            gen_G.train()
            gen_F.train()
            optimizer_G.zero_grad()
            
            # Identity Loss
            id_loss_G = id_loss(gen_G(photo), photo)
            id_loss_F = id_loss(gen_F(monet), monet)
            id_loss_avg = (id_loss_G + id_loss_F)/2
            
            # GAN Loss
            generated_monet = gen_G(photo)
            generated_photo = gen_F(monet)
            gan_loss_G = gan_loss(dY(generated_monet), valid)
            gan_loss_F = gan_loss(dX(generated_photo), valid)
            gan_loss_avg = (gan_loss_G + gan_loss_F)/2
            
            # Cycle Consistency Loss
            cycle_loss_G = cycle_loss(gen_F(generated_monet), photo)
            cycle_loss_F = cycle_loss(gen_G(generated_photo), monet)
            cycle_loss_avg = (cycle_loss_G + cycle_loss_F)/2

            generator_loss = gan_loss_avg + lambda_id * id_loss_avg + lambda_cycle * cycle_loss_avg
            generator_loss.backward()
            optimizer_G.step()

            # TRAIN DISCRIMINATOR X
            optimizer_dX.zero_grad()
            real_loss = gan_loss(dX(photo), valid)
            generated_photo_ = buffer_X.push_and_pop(generated_photo)
            generated_loss = gan_loss(dX(generated_photo_), generated)
            dX_loss = (real_loss + generated_loss)/2
            dX_loss.backward()
            optimizer_dX.step()

            # TRAIN DISCRIMINATOR Y
            optimizer_dY.zero_grad()
            real_loss = gan_loss(dY(monet), valid)
            generated_monet_ = buffer_Y.push_and_pop(generated_monet)
            generated_loss = gan_loss(dY(generated_monet_), generated)
            dY_loss = (real_loss + generated_loss)/2
            dY_loss.backward()
            optimizer_dY.step()

            d_loss = (dX_loss + dY_loss)/2

            print(f'[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [Discriminator Loss {d_loss.item()}] [Generator Loss {generator_loss.item()}]')

            if (epoch + 1) % 5 == 1 and i == 62:
                generated_monet = generated_monet/2 + 0.5
                photo = photo/2 + 0.5
                generated_monet = np.transpose(generated_monet.detach().cpu().numpy()[0, :, :, :])
                photo = np.transpose(photo.detach().cpu().numpy()[0, :, :, :])
                plt.imshow(generated_monet)
                plt.show()
                plt.imshow(photo)
                plt.show()

        torch.save(gen_G.state_dict(), "generator_G")
        torch.save(gen_F.state_dict(), "generator_F")
