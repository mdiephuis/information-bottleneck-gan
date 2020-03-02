import torch.nn as nn
import torch
import numpy as np


def conv_size(H_in, k_size, stride, padd, dil=1):
    H_out = np.floor((H_in + 2 * padd - dil * (k_size - 1) - 1) / stride + 1)
    return np.int(H_out)


class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BatchReshape(nn.Module):
    def __init__(self, shape):
        super(BatchReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class DCGAN_Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(DCGAN_Discriminator, self).__init__()
        self.in_channels = in_channels

        self.network = nn.ModuleList([
            nn.Conv2d(self.in_channels, 32, kernel_size=5, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            BatchFlatten(),
            nn.Linear(64 * 16, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(512, 1)
        ])

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return torch.sigmoid(x)


class DCGAN_Encoder(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(DCGAN_Encoder, self).__init__()

        H_conv_out = conv_size(input_shape[-1], 4, 2, 1)
        H_conv_out = conv_size(H_conv_out, 4, 2, 1)
        convnet_out = np.int(H_conv_out * H_conv_out * out_channels * 2)
        self.H_conv_out = H_conv_out

        self.network = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.Conv2d(1, out_channels, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels * 2, 4, 2, padding=1),
            nn.LeakyReLU(),
            BatchFlatten(),
            nn.Linear(convnet_out, encoder_size),
            nn.LeakyReLU(),
        )

        self.encoder_mu = nn.Linear(encoder_size, latent_size)
        self.encoder_std = nn.Linear(encoder_size, latent_size)

    def encode(self, x):
        x = self.network(x)
        mu = self.encoder_mu(x)
        log_var = self.encoder_std(x)
        log_var = torch.clamp(torch.sigmoid(log_var), min=0.01)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class DCGAN_Generator(nn.Module):
    def __init__(self, H_conv_out, out_channels, decoder_size, latent_size):
        super(DCGAN_Generator, self).__init__()
        self.decoder = nn.ModuleList([
            nn.Linear(latent_size, decoder_size),
            nn.ReLU(),
            nn.Linear(decoder_size, H_conv_out * H_conv_out * out_channels * 2),
            nn.ReLU(),
            BatchReshape((-1, out_channels * 2, H_conv_out, H_conv_out, )),
            nn.ConvTranspose2d(out_channels * 2, out_channels, 4, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, 1, 4, 2, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x


class DCGAN2_Encoder(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(DCGAN2_Encoder, self).__init__()

        H_conv_out = conv_size(input_shape[-1], 4, 2, 1)
        H_conv_out = conv_size(H_conv_out, 3, 1, 1)
        H_conv_out = conv_size(H_conv_out, 4, 2, 1)
        H_conv_out = conv_size(H_conv_out, 3, 1, 1)

        convnet_out = np.int(H_conv_out * H_conv_out * out_channels * 2)

        self.H_conv_out = H_conv_out

        self.network = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.Conv2d(1, out_channels, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels * 2, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels * 2),

            nn.Conv2d(out_channels * 2, out_channels * 2, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels * 2),

            BatchFlatten(),
            nn.Linear(convnet_out, encoder_size)
        )

        self.encoder_mu = nn.Linear(encoder_size, latent_size)
        self.encoder_std = nn.Linear(encoder_size, latent_size)

    def encode(self, x):
        x = self.network(x)
        mu = self.encoder_mu(x)
        log_var = self.encoder_std(x)
        log_var = torch.clamp(torch.sigmoid(log_var), min=0.01)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class DCGAN2_Generator(nn.Module):
    def __init__(self, H_conv_out, out_channels, decoder_size, latent_size):
        super(DCGAN2_Generator, self).__init__()

        self.decoder = nn.ModuleList([
            nn.Linear(latent_size, decoder_size),
            nn.ReLU(),

            nn.Linear(decoder_size, H_conv_out * H_conv_out * out_channels * 2),
            nn.ReLU(),
            BatchReshape((-1, out_channels * 2, H_conv_out, H_conv_out, )),

            nn.ConvTranspose2d(out_channels * 2, out_channels, 4, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels, out_channels // 2, 4, 2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels // 2, 1, 3, 1, padding=1),
        ])

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return torch.tanh(x) * 0.5 + 0.5


class MNIST_Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(MNIST_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x


class MNIST_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MNIST_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.encoder_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder_std = nn.Linear(self.hidden_dim, self.latent_dim)

    def encode(self, x):
        x = self.network(x)
        mu = self.encoder_mu(x)
        log_var = self.encoder_std(x)
        log_var = torch.clamp(torch.sigmoid(log_var), min=0.01)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class MNIST_Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MNIST_Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.network = nn.ModuleList([
            BatchFlatten(),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ])

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return torch.sigmoid(x)

