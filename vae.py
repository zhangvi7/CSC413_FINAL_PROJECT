###########################################################
# CSC413 Project - Contains VAE models and training loops
###########################################################

import torch 
import torch.nn as nn
from torchvision import datasets, transforms, utils
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
import random

# setup device + SEED
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


#fully connected VAE model
class VAE(nn.Module):
    def __init__(self, inp_dim, latent_dim, h_dim):
        super(VAE, self).__init__()

        #inp_dim is flattened image size
        self.inp_dim = inp_dim
        self.latent_dim = latent_dim
        self.h_dim = h_dim

        #### ENCODER ##########
        #3 hidden layer encoder
        self.e_fc1 = nn.Linear(inp_dim, h_dim)
        self.e_fc2 =  nn.Linear(h_dim, int(h_dim/2))
        
        self.e_fc_mean = nn.Linear(int(h_dim/2), latent_dim)
        self.e_fc_log_std = nn.Linear(int(h_dim/2), latent_dim)

        #### DECODER ##########
        # 3 hidden layer decoder
        self.d_fc1 = nn.Linear(latent_dim, int(h_dim/2))
        self.d_fc2 = nn.Linear(int(h_dim/2), h_dim)
        self.d_fc3 = nn.Linear(h_dim, inp_dim)
        
    
    def forward(self, x):
        mean, log_std = self.encode(x)
        z = self.sample(mean, log_std)
        x_hat = self.decode(z)

        return x_hat, mean, log_std

    #given an x from the dataset, finds mean and log_var
    #for the variational distribution q(z)
    def encode(self, x):
        w = F.relu(self.e_fc1(x.view(-1, self.inp_dim)))
        w = F.relu(self.e_fc2(w))
        mean = self.e_fc_mean(w)
        log_std = self.e_fc_log_std(w)

        #batch of mean and log_var for q(z | x)
        return mean, log_std

    def decode(self, z):
        w = F.relu(self.d_fc1(z))
        w = F.relu(self.d_fc2(w))
        
        #image approximation, with pixel values in [0,1]
        #assumed bernoulli distribution over each pixel, and the pixel value
        # is prob(pixel is on)
        x_hat = F.sigmoid(self.d_fc3(w)) 

        return x_hat
    
    #sample using reparameterization trick
    def sample(self, mean, log_std):
        #sample from N(0,I) and use 
        #reparameterization trick to transform
        #to given mean and variance
        std = torch.exp(log_std)

        sample_z = mean + std*torch.randn_like(mean)

        return sample_z

    #if train, samples highest probability values
    def sample_ims(self, num_ims, mean, log_std, train = True):
        if train:
            z = self.sample(mean, log_std)
        else:
            z = mean

        ims = self.decode(z)
        return ims

        
    #sample from the 
    def sample_ims_from_prior(self, num_ims, train, im_size = (28,28)):
        #mean = 0, std = I
        mean = torch.zeros((num_ims, self.latent_dim))
        log_std = torch.zeros((num_ims, self.latent_dim))

        ims_from_prior = self.sample_ims(num_ims, mean, log_std, train)
        ims = ims_from_prior.view((-1,28,28))

        return ims

    def sample_ims_like(self, x, num_ims, train, im_size = (28,28)):
        #mean = 0, std = I
        mean, log_std = self.encode(x)
        ims_from_prior = self.sample_ims(num_ims, mean, log_std, train)
        ims = ims_from_prior.view((-1,28,28))

        return ims
    
def sample_ims_like(vae_model, x, num_ims, train=True, im_size = (28,28)):

    mean, log_std = vae_model.encode(x)

    mean = mean.repeat(num_ims,1)
    log_std = log_std.repeat(num_ims, 1)

    ims = vae_model.sample_ims(num_ims, mean, log_std, train)
    ims = ims.view((-1,28,28))

    return ims


###########################
# Training Loop
###########################
# negative ELBO loss
# = reconstruction_loss + KL(q || z_prior) 
def vae_loss_func(x_hat, x, mean, log_std, batch_size):
    recon_loss = F.binary_cross_entropy(x_hat, x)

    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + 2*log_std - mean**2 - torch.exp(log_std)**2)
    # Normalise by same number of elements as in reconstruction
    KLD /= (batch_size * 784)

    loss = recon_loss+ KLD

    return loss

def vae_val_loss(vae_model, val_loader, batch_size):
    val_loss = 0.0
    batch_size = 0

    for x_batch, labels in val_loader:
    
        x_hat_batch, mean_batch, log_std_batch = vae_model(x_batch)

        loss_value = vae_loss_func(x_hat_batch, x_batch.view(-1, vae_model.inp_dim), mean_batch, log_std_batch, batch_size)

        batch_size += 1
        val_loss += loss_value

    val_loss = val_loss/batch_size

    return val_loss

#example config
#config = {
#    "epoch": 10,
#    "lr": 1e-3,
#    "batch": 100
#}
#config = dict of 'epoch', 'lr', 'batch', ''
def vae_train(vae_model, config, train_dataset, val_dataset):
    vae_model = vae_model.to(device)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config["batch"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = config["batch"], shuffle =False)

    optimizer = torch.optim.Adam(vae_model.parameters(), lr = config["lr"])

    num_epochs = config["epoch"]

    epoch_loss = torch.zeros(num_epochs)
    val_epoch_loss = torch.zeros(num_epochs)

    for epoch in range(num_epochs):

        epoch_loss[epoch] = 0.0
        batch_size = 0

        for x_batch, labels in train_loader:
            
            x_hat_batch, mean_batch, log_std_batch = vae_model(x_batch)

            loss_val = vae_loss_func(x_hat_batch, x_batch.view(-1, vae_model.inp_dim), mean_batch, log_std_batch, config["batch"])

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            batch_size += 1
            epoch_loss[epoch] += loss_val

        epoch_loss[epoch] = epoch_loss[epoch]/batch_size

        #compute val_loss
        val_epoch_loss[epoch] = vae_val_loss(vae_model, val_loader, config["batch"])
        
        print("Epoch {}/{} -- Train loss: {}, Val loss: {}".format(epoch+1, num_epochs, epoch_loss[epoch], val_epoch_loss[epoch]))
    
    plt.plot(epoch_loss.to('cpu').detach().numpy(), label = "Train Loss (-ELBO)")
    plt.plot(val_epoch_loss.to('cpu').detach().numpy(), label="Validation Loss (-ELBO)")
    plt.title("VAE Loss over Epochs") 
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO Loss")
    
#########################
#Convolutional VAE model
##########################
class VAEConv(nn.Module):
    #architecture follows that of the B-VAE paper
    def __init__(self, latent_dim, im_size=(64,64), n_channels=3):
        super(VAEConv, self).__init__()

        #inp_dim is flattened image size
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.n_channels = n_channels

        #### ENCODER ##########
        #Convolution based encoder
        #NOTE im_size must be (64, 64) to work (or a multiple of 64,64)
        self.e_conv1 = nn.Conv2d(in_channels = n_channels,
                                out_channels = 32,
                                 kernel_size =4,
                                 stride=2,
                                 padding=1)
        self.e_conv2 =  nn.Conv2d(in_channels = 32,
                                out_channels = 32,
                                 kernel_size =4,
                                 stride=2,
                                 padding=1)
        self.e_conv3=  nn.Conv2d(in_channels = 32,
                                out_channels = 64,
                                 kernel_size =4,
                                 stride=2,
                                 padding=1)
        self.e_conv4 = nn.Conv2d(in_channels = 64,
                                out_channels = 64,
                                 kernel_size =4,
                                 stride=2,
                                 padding=1)
        #output of this block will be scalar Bx256x(1x1)
        #for input size 64 x 64 - we use fc layer on it next
        self.e_conv5 = nn.Conv2d(in_channels = 64,
                                out_channels = 256,
                                 kernel_size =4,
                                 stride=1)
        #reshape to (B x 256)
        #takes output of convolution
        self.e_fc_mean = nn.Linear(256, latent_dim)
        self.e_fc_log_std = nn.Linear(256, latent_dim)

        #### DECODER ##########
        # Convolutional decoder
        #takes a latent sample z and generates image
        self.d_fc1 = nn.Linear(latent_dim, 256)

        #must reshape to (Bx 1 x 1) next for conv layers
        self.d_conv1 = nn.ConvTranspose2d(in_channels=256,
                                          out_channels=64,
                                          kernel_size=4
                                          )
        self.d_conv2 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=64,
                                          kernel_size=4,
                                          stride=2, 
                                          padding=1)
        self.d_conv3 = nn.ConvTranspose2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2, 
                                    padding=1)
        self.d_conv4 = nn.ConvTranspose2d(in_channels=32,
                            out_channels=32,
                            kernel_size=4,
                            stride=2, 
                            padding=1)
        self.d_conv5 = nn.ConvTranspose2d(in_channels=32,
                    out_channels=n_channels,
                    kernel_size=4,
                    stride=2, 
                    padding=1)
        
    
    def forward(self, x):
        #x inputs expected to be C x H xW
        mean, log_std = self.encode(x)
        z = self.sample(mean, log_std)
        x_hat = self.decode(z)

        return x_hat, mean, log_std

    #given an x from the dataset, finds mean and log_var
    #for the variational distribution q(z)
    def encode(self, x):
        w = F.relu(self.e_conv1(x))
        w = F.relu(self.e_conv2(w))
        w = F.relu(self.e_conv3(w))
        w = F.relu(self.e_conv4(w))
        w = F.relu(self.e_conv5(w))
        
        w = w.view((-1,  256))
        mean = self.e_fc_mean(w)
        log_std = self.e_fc_log_std(w)

        #batch of mean and log_var for q(z | x)
        return mean, log_std

    def decode(self, z):
        w = F.relu(self.d_fc1(z))
        w = w.view((-1, 256, 1, 1))
        w = F.relu(self.d_conv1(w))
        w = F.relu(self.d_conv2(w))
        w = F.relu(self.d_conv3(w))
        w = F.relu(self.d_conv4(w))
        w = self.d_conv5(w)

        #image approximation, with pixel values in [0,1]
        #assumed bernoulli distribution over each pixel, and the pixel value
        # is prob(pixel is on)
        #x_hat = F.sigmoid(self.d_fc3(w)) 
        x_hat = w

        #note: im has channel C X H x W
        return x_hat
    
    #sample using reparameterization trick
    def sample(self, mean, log_std):
        #sample from N(0,I) and use 
        #reparameterization trick to transform
        #to given mean and variance
        std = torch.exp(log_std)

        sample_z = mean + std*torch.randn_like(mean)

        return sample_z

    #if train, samples highest probability values
    def sample_ims(self, num_ims, mean, log_std, train = True):
        if train:
            z = self.sample(mean, log_std)
        else:
            z = mean

        ims = self.decode(z)
        ims = F.sigmoid(ims)
        #print(ims.size())
        return ims.permute(0, 2,3,1)

        
    #sample from the 
    def sample_ims_from_prior(self, num_ims, train=True):
        #mean = 0, std = I
        mean = torch.zeros((num_ims, self.latent_dim))
        log_std = torch.zeros((num_ims, self.latent_dim))

        ims_from_prior = self.sample_ims(num_ims, mean, log_std, train)
        #ims = ims_from_prior.view((-1,self.im_size[0],self.im_size[1]))

        return ims_from_prior

    def sample_ims_like(self, x, num_ims, train=True):
        mean, log_std = self.encode(x)

        mean = mean.repeat(num_ims,1)
        log_std = log_std.repeat(num_ims, 1)

        ims = self.sample_ims(num_ims, mean, log_std, train)
        #ims = ims.view((-1,im_size[0], im_size[1]))

        #place channel at end
        return ims
    
################################
# Training functions for conv VAE
######################################

## train function for conv VAE
# negative ELBO loss
# = reconstruction_loss + KL(q || z_prior) 
#some code from https://github.com/1Konny/Beta-VAE/blob/master/solver.py
def vae_conv_loss_func(x_hat, x, mean, log_std, batch_size, decoder_type="B"):
    if decoder_type=="B":
        #bernoulli type output distribution assumed
        recon_loss = F.binary_cross_entropy_with_logits(x_hat, x, size_average=False).div(batch_size)
    #for Gaussian type output, do MSE 
    else:
        x_hat = F.sigmoid(x_hat)
        recon_loss = F.mse_loss(x_hat, x, size_average=False).div(batch_size)

    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * (1 + 2*log_std - mean**2 - torch.exp(2*log_std)).sum(1).mean(0, True)
    # Normalise by same number of elements as in reconstruction
    #KLD /= (batch_size * 784)

    loss = recon_loss+ KLD

    return loss


#config = dict of 'epoch', 'lr', 'batch', ''
def vae_train_dsprites(vae_model, config, train_imgs, decoder_type="B"):
    vae_model = vae_model.to(device)

    #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config["batch"], shuffle=True)

    optimizer = torch.optim.Adam(vae_model.parameters(), lr = config["lr"])

    num_epochs = config["epoch"]

    epoch_loss = torch.zeros(num_epochs).to(device)

    num_batches = int(np.round(np.shape(train_imgs)[0]/config["batch"]))

    for epoch in range(num_epochs):

        epoch_loss[epoch] = 0.0
        batch_size = 0

        for idx in range(num_batches):
            x_batch = train_imgs[(idx*config["batch"]):((idx+1)*config["batch"])]
            x_batch = torch.tensor(x_batch)
            x_batch = x_batch.float().to(device)
            x_batch = x_batch.view((-1, 1, 64, 64))
            x_hat_batch, mean_batch, log_std_batch = vae_model(x_batch)

            #print(x_hat_batch.size())

            loss_val = vae_conv_loss_func(x_hat_batch, x_batch, 
                                          mean_batch, 
                                          log_std_batch, config["batch"],
                                          decoder_type=decoder_type)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            batch_size += 1
            epoch_loss[epoch] += loss_val.item()

        epoch_loss[epoch] = epoch_loss[epoch]/batch_size

     
        print("Epoch {}/{} -- Train loss: {}".format(epoch+1, num_epochs, epoch_loss[epoch]))
    
    plt.plot(epoch_loss.to('cpu').detach().numpy(), label = "Train Loss (-ELBO)")
    plt.title("VAE Loss over Epochs") 
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO Loss")