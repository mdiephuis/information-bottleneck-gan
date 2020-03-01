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
parser.add_argument('--latent-size', type=int, default=100, metavar='N',
                    help='Noise dimension (default: 10)')
parser.add_argument('--out-channels', type=int, default=64, metavar='N',
                    help='VAE 2D conv channel output (default: 64')
parser.add_argument('--encoder-size', type=int, default=1024, metavar='N',
                    help='VAE encoder size (default: 1024')
parser.add_argument('--learning-rate', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4')
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
    device = torch.device("cuda")
    torch.cuda.set_device(0)
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


def train_validate(E, G, D, EG_optim, G_optim, D_optim, loader, epoch, is_train):

    img_shape = loader.img_shape

    data_loader = loader.train_loader if is_train else loader.test_loader

    E.train() if is_train else E.eval()
    G.train() if is_train else G.eval()
    D.train() if is_train else D.eval()

    vae_batch_loss = 0
    generator_batch_loss = 0
    discriminator_batch_loss = 0

    # discriminator score on x and x_hat
    score_dx = 0
    score_d_x_hat = 0

    loss_bce_sum = nn.BCELoss(reduction='sum')

    for batch_idx, (x, _) in enumerate(data_loader):

        batch_size = x.size(0)

        x = x.cuda() if args.cuda else x
        x = x.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        eta = sample_gauss_noise(batch_size, img_shape[1] * img_shape[2], 0, 0.1)

        eta = eta.cuda() if args.cuda else eta

        x += eta.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        # Generator forward

        z_draw = sample_gauss_noise(batch_size, args.latent_size)
        z_draw = z_draw.cuda() if args.cuda else z_draw

        #############################################
        # Encoder forward
        # z_hat, _, _ = E(x)
        # z_hat = z_hat.detach()

        x_hat = G(z_draw)
        y_hat = D(x_hat.view(batch_size, img_shape[0], img_shape[1], img_shape[2]))

        # Real data, discriminator forward
        y_real = D(x)

        # Discriminator loss
        y_ones = torch.ones(batch_size, 1)
        y_zeros = torch.zeros(batch_size, 1)

        y_ones = y_ones.cuda() if args.cuda else y_ones
        y_zeros = y_zeros.cuda() if args.cuda else y_zeros
        #
        score_dx += y_real.data.mean()
        score_d_x_hat += y_hat.data.mean()

        #############################################
        # Discriminator loss
        discriminator_loss = loss_bce_sum(y_real, y_ones) + loss_bce_sum(y_hat, y_zeros)

        discriminator_batch_loss += discriminator_loss.item() / batch_size

        if is_train:
            D_optim.zero_grad()
            D_optim.step()

        #############################################
        # Generator update

        z_draw = sample_gauss_noise(batch_size, args.latent_size)
        z_draw = z_draw.cuda() if args.cuda else z_draw

        # Generator forward
        x_hat = G(z_draw)
        y_hat = D(x_hat.view(batch_size, img_shape[0], img_shape[1], img_shape[2]))

        generator_loss = loss_bce_sum(y_hat, y_ones) + loss_bce_sum(x_hat, x)

        generator_batch_loss += generator_loss.item() / batch_size

        if is_train:
            G_optim.zero_grad()
            generator_loss.backward(retain_graph=True)
            G_optim.step()

        #############################################
        # Encoder - Generator update
        z_hat, z_mu, z_logvar = E(x)

        # Generator forward
        x_hat = G(z_hat)

        # Loss 1, kl divergence
        loss_kld = loss_kl_gauss(z_mu, z_logvar)

        # Loss 2, reconstruction loss
        loss_recon = loss_bce_sum(x_hat.view(-1, 1), x.view(-1, 1))

        vae_loss = loss_kld + loss_recon
        vae_batch_loss += vae_loss.item() / batch_size

        if is_train:
            EG_optim.zero_grad()
            vae_loss.backward(retain_graph=True)
            EG_optim.step()

    print('D(x): %.4f D(G(z)): %.4f , %.4f' % (score_dx / (batch_idx + 1), score_d_x_hat / (batch_idx + 1)))

    return vae_batch_loss / (batch_idx + 1), generator_batch_loss / (batch_idx + 1), discriminator_batch_loss / (batch_idx + 1)


def execute_graph(E, G, D, EG_optim, G_optim, D_optim, loader, epoch, use_tb):
    print('=> epoch: {}'.format(epoch))
    # Training loss
    VAE_t_loss, G_t_loss, D_t_loss = train_validate(E, G, D, EG_optim, G_optim, D_optim, loader, epoch, is_train=True)

    # Validation loss
    VAE_v_loss, G_v_loss, D_v_loss = train_validate(E, G, D, EG_optim, G_optim, D_optim, loader, epoch, is_train=False)

    print('=> epoch: {} Average Train VAE loss: {:.4f}, G loss: {:.4f}, D loss: {:.4f}'.format(epoch, VAE_t_loss, G_t_loss, D_t_loss))
    print('=> epoch: {} Average Valid VAE loss: {:.4f}, G loss: {:.4f}, D loss: {:.4f}'.format(epoch, VAE_v_loss, G_v_loss, D_v_loss))

    if use_tb:
        logger.add_scalar(log_dir + '/VAE-train-loss', VAE_t_loss, epoch)
        logger.add_scalar(log_dir + '/G-train-loss', G_t_loss, epoch)
        logger.add_scalar(log_dir + '/D-train-loss', D_t_loss, epoch)

        logger.add_scalar(log_dir + '/VAE-valid-loss', VAE_v_loss, epoch)
        logger.add_scalar(log_dir + '/G-valid-loss', G_v_loss, epoch)
        logger.add_scalar(log_dir + '/D-valid-loss', D_v_loss, epoch)

    # Generate examples
        img_shape = loader.img_shape
        sample = dcgan_generation_example(G, args.latent_size, 10, img_shape, args.cuda)
        sample = sample.detach()
        sample = tvu.make_grid(sample, normalize=True, scale_each=True)
        logger.add_image('generation example', sample, epoch)

    # Reconstruction example
        reconstructed = dcgan_reconstruction_example(E, G, loader.test_loader, 10, img_shape, args.cuda)
        reconstructed = reconstructed.detach()
        reconstructed = tvu.make_grid(reconstructed, normalize=True, scale_each=True)
        logger.add_image('reconstruction example', reconstructed, epoch)

    # Manifold example
    if args.latent_size == 2:
        sample = manifold_generation_example(G, img_shape, epoch, args.cuda)
        sample = sample.detach()
        sample = tvu.make_grid(sample, normalize=True, scale_each=True)
        logger.add_image('manifold example', sample, epoch)

    return G_v_loss, D_v_loss


# MNIST Model definitions
encoder_size = args.encoder_size
decoder_size = args.encoder_size
latent_size = args.latent_size
out_channels = args.out_channels
in_channels = loader.img_shape[0]

E = DCGAN2_Encoder(loader.img_shape, out_channels, encoder_size, latent_size).type(dtype)
h_conv_outsize = E.H_conv_out
print(E)

G = DCGAN2_Generator(h_conv_outsize, out_channels, decoder_size, latent_size).type(dtype)

print(G)

D = DCGAN_Discriminator(in_channels).type(dtype)
# D = MNIST_Discriminator(784, 500).type(dtype)
print(D)


E.apply(init_xavier_weights)
G.apply(init_xavier_weights)
D.apply(init_xavier_weights)


beta1 = 0.5
beta2 = 0.999

# E_optim = torch.optim.RMSprop(E.parameters(), lr=1e-3, weight_decay=1e-5)
E_optim = torch.optim.Adam(E.parameters(), lr=1e-3, betas=(beta1, beta2))
G_optim = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(beta1, beta2))

EG_optim = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=1e-3, betas=(beta1, beta2))

D_optim = torch.optim.Adam(D.parameters(), lr=1e-5, betas=(beta1, beta2))


# Main training loop
for epoch in range(1, args.epochs):
    _, _ = execute_graph(E, G, D, EG_optim, G_optim, D_optim, loader, epoch, use_tb)


# TensorboardX logger
logger.close()
