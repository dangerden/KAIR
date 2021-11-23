# Inspired by Improving Image Autoencoder Embeddings with Perceptual Loss
# https://github.com/guspih/Perceptual-Autoencoders
import random
import numpy as np
import datetime
import time
import sys
import os
import matplotlib.pyplot as plt
import torchvision
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from tqdm import tqdm
from torchsummary import summary

def _create_coder(channels, kernel_sizes, strides, conv_types,
    activation_types, paddings=(0,0), out_paddings=None, batch_norms=False):
    '''
    Function that creates en- or decoders based on parameters
    Args:
        channels ([int]): Channel sizes per layer. 1 more than layers
        kernel_sizes ([int]): Kernel sizes per layer
        strides ([int]): Strides per layer
        conv_types ([f()->type]): Type of the convoultion module per layer
        activation_types ([f()->type]): Type of activation function per layer
        paddings ([(int, int)]): The padding per layer
        batch_norms ([bool]): Whether to use batchnorm on each layer
    Returns (nn.Sequential): The created coder
    '''
    if not isinstance(conv_types, list):
        conv_types = [conv_types for _ in range(len(kernel_sizes))]

    if not isinstance(activation_types, list):
        activation_types = [activation_types for _ in range(len(kernel_sizes))]

    if not isinstance(paddings, list):
        paddings = [paddings for _ in range(len(kernel_sizes))]

    if not out_paddings is None and not isinstance(out_paddings, list):
        out_paddings = [out_paddings for _ in range(len(kernel_sizes))]
        
    if not isinstance(batch_norms, list):
        batch_norms = [batch_norms for _ in range(len(kernel_sizes))]

    coder = nn.Sequential()
    for layer in range(len(channels)-1):
        if out_paddings is None:
          coder.add_module(
              'conv'+ str(layer), 
              conv_types[layer](
                  in_channels=channels[layer], 
                  out_channels=channels[layer+1],
                  kernel_size=kernel_sizes[layer],
                  stride=strides[layer],
                  padding=paddings[layer]
              )
          )
        else:
          coder.add_module(
              'conv'+ str(layer), 
              conv_types[layer](
                  in_channels=channels[layer], 
                  out_channels=channels[layer+1],
                  kernel_size=kernel_sizes[layer],
                  stride=strides[layer],
                  padding=paddings[layer],
                  output_padding=out_paddings[layer]
              )
          )
        if batch_norms[layer]:
            coder.add_module(
                'norm'+str(layer),
                nn.BatchNorm2d(channels[layer+1])
            )
        if not activation_types[layer] is None:
            coder.add_module('acti'+str(layer),activation_types[layer]())

    return coder

class TemplateVAE(nn.Module):
    '''
    A template class for Variational Autoencoders to minimize code duplication
    Args:
        input_size (int,int): The height and width of the input image
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use (None for pixel-wise)
    '''
    
    def __str__(self):
        string = super().__str__()[:-1]
        string = string + '  (variational): {}\n  (gamma): {}\n)'.format(
                self.variational,self.gamma
            )
        return string

    def __repr__(self):
        string = super().__repr__()[:-1]
        string = string + '  (variational): {}\n  (gamma): {}\n)'.format(
                self.variational,self.gamma
            )
        return string
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        out = eps.mul(std).add_(mu)
        return out

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.variational:
            z = self.sample(mu, logvar)
        else:
            z = mu
        rec_x = self.decode(z)
        return rec_x, z, mu, logvar

    def loss(self, output, x):
        rec_x, z, mu, logvar = output
        if self.perceptual_loss:
            x = self.perceptual_net(x)
            rec_x = self.perceptual_net(rec_x)
        else:
            x = x.reshape(x.size(0), -1)
            rec_x = rec_x.view(x.size(0), -1)
        REC = F.mse_loss(rec_x, x, reduction='mean')

        if self.variational:
            KLD = -1 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return REC + self.gamma*KLD, REC, KLD
        else:
            return [REC]

class FourLayerCVAE(TemplateVAE):
    '''
    A Convolutional Variational Autoencoder for images
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use (None for pixel-wise)
    '''

    def __init__(self, input_size=(64,64), z_dimensions=32,
        variational=True, gamma=20.0, perceptual_net=None
    ):
        super().__init__()

        #Parameter check
        if (input_size[0] - 64) % 16 != 0 or (input_size[1] - 64) % 16 != 0:
            raise ValueError(f'Input_size is {input_size}, but must be 64+16*N')

        #Attributes
        self.input_size = input_size
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_net = perceptual_net

        self.perceptual_loss = not perceptual_net is None
            
        encoder_channels = [3,32,64,128,256]
        self.encoder = _create_coder(
            encoder_channels, [4,4,4,4], [2,2,2,2],
            nn.Conv2d, nn.ReLU,
            batch_norms=[True,True,True,True]
        )
        
        f = lambda x: np.floor((x - (2,2))/2)
        conv_sizes = f(f(f(f(np.array(input_size)))))
        conv_flat_size = int(encoder_channels[-1]*conv_sizes[0]*conv_sizes[1])
        self.mu = nn.Linear(conv_flat_size, self.z_dimensions)
        self.logvar = nn.Linear(conv_flat_size, self.z_dimensions)
        
        g = lambda x: int((x-64)/16)+1
        deconv_flat_size = g(input_size[0]) * g(input_size[1]) * 1024
        self.dense = nn.Linear(self.z_dimensions, deconv_flat_size)
        
        self.decoder = _create_coder(
            [1024,128,64,32,3], [5,5,6,6], [2,2,2,2],
            nn.ConvTranspose2d,
            [nn.ReLU,nn.ReLU,nn.ReLU,nn.Sigmoid],
            batch_norms=[True,True,True,False]
        )

        self.relu = nn.ReLU()

    def decode(self, z):
        y = self.dense(z)
        y = self.relu(y)
        y = y.view(
            y.size(0), 1024,
            int((self.input_size[0]-64)/16)+1,
            int((self.input_size[1]-64)/16)+1
        )
        y = self.decoder(y)
        return y


class ThreeLayerCVAE_v1(TemplateVAE):
    '''
    A Convolutional Variational Autoencoder for images
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use (None for pixel-wise)
    '''

    def __init__(self, input_size=(64,64), z_dimensions=16,
        variational=True, gamma=20.0, perceptual_net=None
    ):
        super().__init__()

        #Parameter check
        if (input_size[0] - 64) % 16 != 0 or (input_size[1] - 64) % 16 != 0:
            raise ValueError(f'Input_size is {input_size}, but must be 64+16*N')

        #Attributes
        self.input_size = input_size
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_net = perceptual_net

        self.perceptual_loss = not perceptual_net is None
            
        encoder_channels = [3,16,64,256]
        self.encoder = _create_coder(
            encoder_channels, [3,3,3], [2,2,2],
            nn.Conv2d, nn.LeakyReLU,
            paddings=[(1,1),(1,1),(1,1)],
            batch_norms=[True,True,True]
        )
        
        f = lambda x: np.floor((x - (2,2))/2+1)
        conv_sizes = f(f(f(np.array(input_size))))
        print(f"conv_sizes: {conv_sizes}")
        conv_flat_size = int(encoder_channels[-1]*conv_sizes[0]*conv_sizes[1])
        print(f"conv_flat_sizes: {conv_flat_size}")
        #self.mu = nn.Linear(conv_flat_size, self.z_dimensions)
        #self.logvar = nn.Linear(conv_flat_size, self.z_dimensions)
        
        g = lambda x: int((x)/8)
        print(f"g_size: {g(input_size[0])}")
        deconv_flat_size = g(input_size[0]) * g(input_size[1]) * 32
        print(f"deconv_flat_size: {deconv_flat_size}")
        #self.dense = nn.Linear(self.z_dimensions, deconv_flat_size)
        
        neck_ch = encoder_channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(neck_ch,z_dimensions,3,padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(z_dimensions,neck_ch,3,padding='same'),
            nn.LeakyReLU()
        )

        self.decoder = _create_coder(
            [256,64,16,3], [3,3,3], [2,2,2],
            nn.ConvTranspose2d,
            [nn.ReLU,nn.ReLU,nn.Sigmoid],
            paddings=[(1,1),(1,1),(1,1)],
            out_paddings=[(1,1),(1,1),(1,1)],
            batch_norms=[True,True,False,False]
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.encoder(x)
        flat_x = x.view(x.size(0),-1)
        mu = None#self.mu(flat_x)
        logvar = None#self.logvar(flat_x)
        return mu, logvar, x

    def decode(self, z):
        #y = self.dense(z)
        #y = self.relu(y)
        #y = y.view(
        #    y.size(0), 32,
        #    int((self.input_size[0])/8),
        #    int((self.input_size[1])/8)
        #)
        #y = self.decoder(y)
        y = self.decoder(z)
        return y

    def forward(self, x):
        mu, logvar, src = self.encode(x)
        if self.variational:
            z = self.sample(mu, logvar)
        else:
            z = mu
        #rec_x = self.decode(z)
        src = self.bottleneck(src)
        rec_x = self.decode(src)
        return rec_x, z, mu, logvar

class FourLayerCVAE(TemplateVAE):
    '''
    A Convolutional Variational Autoencoder for images
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use (None for pixel-wise)
    '''

    def __init__(self, input_size=(64,64), z_dimensions=32,
        variational=True, gamma=20.0, perceptual_net=None
    ):
        super().__init__()

        #Parameter check
        if (input_size[0] - 64) % 16 != 0 or (input_size[1] - 64) % 16 != 0:
            raise ValueError(f'Input_size is {input_size}, but must be 64+16*N')

        #Attributes
        self.input_size = input_size
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_net = perceptual_net

        self.perceptual_loss = not perceptual_net is None
            
        encoder_channels = [3,32,64,128,256]
        self.encoder = _create_coder(
            encoder_channels, [4,4,4,4], [2,2,2,2],
            nn.Conv2d, nn.ReLU,
            batch_norms=[True,True,True,True]
        )
        
        f = lambda x: np.floor((x - (2,2))/2)
        conv_sizes = f(f(f(f(np.array(input_size)))))
        conv_flat_size = int(encoder_channels[-1]*conv_sizes[0]*conv_sizes[1])
        self.mu = nn.Linear(conv_flat_size, self.z_dimensions)
        self.logvar = nn.Linear(conv_flat_size, self.z_dimensions)
        
        g = lambda x: int((x-64)/16)+1
        deconv_flat_size = g(input_size[0]) * g(input_size[1]) * 1024
        self.dense = nn.Linear(self.z_dimensions, deconv_flat_size)
        
        self.decoder = _create_coder(
            [1024,128,64,32,3], [5,5,6,6], [2,2,2,2],
            nn.ConvTranspose2d,
            [nn.ReLU,nn.ReLU,nn.ReLU,nn.Sigmoid],
            batch_norms=[True,True,True,False]
        )

        self.relu = nn.ReLU()

    def decode(self, z):
        y = self.dense(z)
        y = self.relu(y)
        y = y.view(
            y.size(0), 1024,
            int((self.input_size[0]-64)/16)+1,
            int((self.input_size[1]-64)/16)+1
        )
        y = self.decoder(y)
        return y

def show(imgs, block=False, save=None, heading='Figure', fig_axs=None, torchy=True):
    '''
    Paints a column of torch images
    Args:
        imgs ([3darray]): Array of images in shape (channels, width, height)
        block (bool): Whether the image should interupt program flow
        save (str / None): Path to save the image under. Will not save if None
        heading (str)): The heading to put on the image
        fig_axs (plt.Figure, axes.Axes): Figure and Axes to paint on
    Returns (plt.Figure, axes.Axes): The Figure and Axes that was painted
    '''
    if fig_axs is None:
        fig, axs = plt.subplots(1,len(imgs))
        if len(imgs) == 1:
            axs = [axs]
    else:
        fig, axs = fig_axs
        plt.figure(fig.number)
    fig.canvas.set_window_title(heading)
    for i, img in enumerate(imgs):
        if torchy:
            img = img[0].detach().permute(1,2,0)
        plt.axes(axs[i])
        plt.imshow(img)
    plt.show(block=block)
    plt.pause(0.001)
    if not save is None:
        plt.savefig(save)
    return fig, axs

def show_recreation(dataset, m, block=False, save=None):
    '''
    Shows a random image and the encoders attempted recreation
    Args:
        dataset (data.Dataset): Torch Dataset with the image data
        m (nn.Module): (V)AE model to be run
        block (bool): Whether to stop execution until user closes image
        save (str / None): Path to save the image under. Will not save if None
    '''
    with torch.no_grad():
      #img1 = dataset[random.randint(0,len(dataset)-1)].unsqueeze(0)
      img1 = dataset[0].unsqueeze(0)
      if next(m.parameters()).is_cuda:
        img1 = img1.cuda()
      img2, z, mu, logvar = m(img1)
      ev_loss = m.loss((img2, z, mu, logvar), img1)
    show([img1.cpu(),img2.cpu()], block=block, save=save,heading='Random image recreation')    
    print(f"Evaluated loss: {ev_loss[0].item():.4f}")

if __name__ == '__main__':

  model = ThreeLayerCVAE(input_size=(512,512), variational=False).cuda().train()
  summary(model, (3,512,512))

  drive_root ='G:/My Drive'
  model_dir = f'{drive_root}/sr-sat/ae'
  data_dir = f'C:/Users/denis/PeakVisor Dropbox/SilhouettesML/super_res/train_64_512/hr_512'
  val_dir = f'{drive_root}/sr-sat/KAIR/testsets/set5/HR'

  class SRImageDataset(Dataset):
    def __init__(self, img_dir):
      self.img_dir = img_dir
      self.images = [name for name in os.listdir(self.img_dir)]
      self.transform = torch.nn.Sequential(
        torchvision.transforms.ConvertImageDtype(torch.float32),
      )
      self.scripted_transforms = torch.jit.script(self.transform)

    def __len__(self):
      return len(self.images)

    def __getitem__(self, idx):
      try:
        img = torchvision.io.read_image(os.path.join(self.img_dir, self.images[idx]))
        img = self.scripted_transforms(img)
      except:
        return None
      return img

  sr_data_set = SRImageDataset(data_dir)
  val_data_set = SRImageDataset(val_dir)

  print(f'Loading {len(sr_data_set)} train photos')

  def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

  train_dataloader = DataLoader(sr_data_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
  val_dataloader = DataLoader(val_data_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

  num_epochs = 0
  batch_size = 2048
  learning_rate = 3e-4
  display_ep = 1
  save_ep = 10


  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
  total_loss = 0
  start_time = time.time()
  for epoch in range(num_epochs):
      model.train()
      total_loss = 0
      for data in tqdm(train_dataloader):
        img = data.cuda()
        # ===================forward=====================
        output = model(img)
        loss = model.loss(output, img)
        loss[0].backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss[0].item()
      # ===================log========================
      total_loss /= len(train_dataloader)
      print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, total_loss))
      
      #if epoch % display_ep == 0:
        #model.eval()
        #show_recreation(val_data_set, model)

      if epoch % save_ep ==0:
        model_path = f'{model_dir}/vae-3090-{epoch}-{total_loss:.4f}.pth'
        print(f"{time.strftime('%X')} Saving model to {model_path}")
      torch.save(model.state_dict(), model_path)