import math
import numpy as np
import torch.nn as nn
from collections import OrderedDict

"""
# --------------------------------------------
# Perceptual Auto Encoder
# Inspired by Improving Image Autoencoder Embeddings with Perceptual Loss
# https://github.com/guspih/Perceptual-Autoencoders
# --------------------------------------------
"""


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

class ThreeLayerCVAE(TemplateVAE):
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

    def __init__(self, input_size=(64,64), z_dimensions=16, variational=True, gamma=20.0, perceptual_net=None):
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
        #self.bottleneck = nn.Sequential(OrderedDict([
        #    ('bneck-conv2d-1', nn.Conv2d(neck_ch,z_dimensions,3,padding='same')),
        #    ('bneck-lrelu-1', nn.LeakyReLU()),
        #    ('bneck-conv2d-2', nn.Conv2d(z_dimensions,neck_ch,3,padding='same')),
        #    ('bneck-lrelu-2', nn.LeakyReLU())
        #]))
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