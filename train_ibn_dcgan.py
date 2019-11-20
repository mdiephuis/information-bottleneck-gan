import argparse
import torch
from tensorboardX import SummaryWriter
import torchvision.utils as tvu

from models import *
from utils import *
from data import *


parser = argparse.ArgumentParser(description='DCGAN')

parser.add_argument('--uid', type=str, default='IBN_DCGAN',
                    help='Staging identifier (default: DCGAN)')
parser.add_argument('--dataset-name', type=str, default='MNIST',
                    help='Name of dataset (default: MNIST')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of training epochs (default: 15)')
parser.add_argument('--latent-dim', type=int, default=10, metavar='N',
                    help='Noise dimension (default: 10)')
parser.add_argument('--out-channels', type=int, default=64, metavar='N',
                    help='VAE 2D conv channel output (default: 64')
parser.add_argument('--encoder-size', type=int, default=1024, metavar='N',
                    help='VAE encoder size (default: 1024')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()

# Set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)

# Enable CUDA, set tensor type and device
if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    torch.cuda.set_device(1)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Data set transforms
transforms = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
transforms = None

# Get train and test loaders for dataset
loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


def train_validate(E, G, D, E_optim, G_optim, D_optim, loader, epoch, is_train):

    img_shape = loader.img_shape

    data_loader = loader.train_loader if is_train else loader.test_loader

    E.train() if is_train else E.eval()
    G.train() if is_train else G.eval()
    D.train() if is_train else D.eval()

    E_batch_loss = 0
    G_batch_loss = 0
    D_batch_loss = 0
    loss_bce = nn.BCELoss(reduction='sum')

    for batch_idx, (x, _) in enumerate(data_loader):

        batch_size = x.size(0)

        x = x.cuda() if args.cuda else x
        x = x.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        eta = sample_gauss_noise(batch_size, img_shape[1] * img_shape[2], 0, 0.1)

        eta = eta.cuda() if args.cuda else eta

        x += eta.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        # Encoder forward
        z_hat, z_mu, z_logvar = E(x.view(batch_size, -1))

        # RRRRROUND 1

        # Generator forward
        x_hat = G(z_hat)
        y_hat = D(x_hat.view(batch_size, img_shape[0], img_shape[1], img_shape[2]))

        # Loss 1, kl divergence
        loss_kld = loss_kl_gauss(z_mu, z_logvar)

        # Loss 2, Reconstruction
        loss_recon = loss_bce(x_hat.view(-1, 1), x.view(-1, 1))

        encoder_loss = loss_kld + loss_recon

        E_batch_loss += encoder_loss.item() / batch_size

        # Real data, discriminator forward
        y_real = D(x)

        # Discriminator loss
        y_ones = torch.ones(batch_size)
        y_zeros = torch.zeros(batch_size)

        y_ones = y_ones.cuda() if args.cuda else y_ones
        y_zeros = y_zeros.cuda() if args.cuda else y_zeros

        # Discriminator loss
        discriminator_loss = loss_bce(y_real, y_ones) + loss_bce(y_hat, y_zeros)

        D_batch_loss += discriminator_loss.item() / batch_size

        if is_train:
            E_optim.zero_grad()
            encoder_loss.backward(retain_graph=True)
            E_optim.step()

            D_optim.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            D_optim.step()

        # RRound 2
        # Encoder forward
        z_hat, _, _ = E(x)
        z_hat = z_hat.detach()

        # Generator forward
        x_hat = G(z_hat)
        y_hat = D(x_hat.view(batch_size, img_shape[0], img_shape[1], img_shape[2]))

        loss_recon = loss_bce(x_hat.view(-1, 1), x.view(-1, 1))

        # Discriminator loss
        y_ones = torch.ones(batch_size)
        y_ones = y_ones.cuda() if args.cuda else y_ones

        # Discriminator loss
        discriminator_loss = loss_bce(y_real, y_ones)

        generator_loss = 1e-6 * loss_recon + discriminator_loss

        G_batch_loss += generator_loss.item() / batch_size

        if is_train:
            G_optim.zero_grad()
            generator_loss.backward()
            G_optim.step()

    return E_batch_loss / (batch_idx + 1), G_batch_loss / (batch_idx + 1), D_batch_loss / (batch_idx + 1)


def execute_graph(E, G, D, E_optim, G_optim, D_optim, loader, epoch, use_tb):

    # Training loss
    E_t_loss, G_t_loss, D_t_loss = train_validate(E, G, D, E_optim, G_optim, D_optim, loader, epoch, is_train=True)

    # Validation loss
    E_v_loss, G_v_loss, D_v_loss = train_validate(E, G, D, E_optim, G_optim, D_optim, loader, epoch, is_train=False)

    print('=> epoch: {} Average Train E loss: {:.4f}, G loss: {:.4f}, D loss: {:.4f}'.format(epoch, E_t_loss, G_t_loss, D_t_loss))
    print('=> epoch: {} Average Valid E loss: {:.4f}, G loss: {:.4f}, D loss: {:.4f}'.format(epoch, E_v_loss, G_v_loss, D_v_loss))

    if use_tb:
        logger.add_scalar(log_dir + '/E-train-loss', E_t_loss, epoch)
        logger.add_scalar(log_dir + '/G-train-loss', G_t_loss, epoch)
        logger.add_scalar(log_dir + '/D-train-loss', D_t_loss, epoch)

        logger.add_scalar(log_dir + '/E-valid-loss', E_v_loss, epoch)
        logger.add_scalar(log_dir + '/G-valid-loss', G_v_loss, epoch)
        logger.add_scalar(log_dir + '/D-valid-loss', D_v_loss, epoch)

    # Generate examples
        img_shape = loader.img_shape
        sample = mnist_generation_example(G, args.latent_dim, 10, img_shape, args.cuda)
        sample = sample.detach()
        sample = tvu.make_grid(sample, normalize=True, scale_each=True)
        logger.add_image('generation example', sample, epoch)

    return G_v_loss, D_v_loss


# MNIST Model definitions
encoder_size = args.encoder_size
decoder_size = args.encoder_size
latent_dim = args.latent_dim
out_channels = args.out_channels
in_channels = loader.img_shape[0]

E = DCGAN_Encoder(loader.img_shape, out_channels, encoder_size, latent_size).type(dtype)
h_conv_outsize = E.H_conv_out
G = DCGAN_Generator(h_conv_outsize, out_channels, decoder_size, latent_size)
D = DCGAN_Discriminator(in_channels)


E.apply(init_xavier_weights)
G.apply(init_xavier_weights)
D.apply(init_xavier_weights)

learning_rate = 1e-3
beta1 = 0.5
beta2 = 0.999

E_optim = torch.optim.Adam(E.parameters(), lr=learning_rate, betas=(beta1, beta2))
G_optim = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))
D_optim = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))


# Main training loop
for epoch in range(1, args.epochs):
    _, _ = execute_graph(E, G, D, E_optim, G_optim, D_optim, loader, epoch, use_tb)


# TensorboardX logger
logger.close()
