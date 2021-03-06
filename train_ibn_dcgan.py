import argparse
import torch
from tensorboardX import SummaryWriter
import torchvision.utils as tvu

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import os
import time

from models import *
from utils import *
from data import *


parser = argparse.ArgumentParser(description='IBN')

parser.add_argument('--uid', type=str, default='IBN_DCGAN',
                    help='Staging identifier (default: DCGAN)')
parser.add_argument('--dataset-name', type=str, default='CelebA',
                    help='Name of dataset (default: CelebA')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of training epochs (default: 15)')
parser.add_argument('--latent-size', type=int, default=64, metavar='N',
                    help='Noise dimension (default: 10)')
parser.add_argument('--out-channels', type=int, default=64, metavar='N',
                    help='VAE 2D conv channel output (default: 64')
parser.add_argument('--encoder-size', type=int, default=1024, metavar='N',
                    help='VAE encoder size (default: 1024')

parser.add_argument('--g-learning-rate', type=float, default=1e-3,
                    help='Generator learning rate (default: 1e-3')
parser.add_argument('--eg-learning-rate', type=float, default=1e-3,
                    help='Encoder-Generator learning rate (default: 1e-3')
parser.add_argument('--d-learning-rate', type=float, default=1e-3,
                    help='Discriminator learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=0.75, action="store", type=float,
                    help='Learning rate decay (default: 0.75')

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

# Setup asset directories
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('runs'):
    os.makedirs('runs')

# Enable CUDA, set tensor type and device
if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(1)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


if args.dataset_name == 'CelebA':
    in_channels = 3

    loader = CelebALoader(args.data_dir, args.batch_size, 0.2, True, True, args.cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader

else:
    transforms = [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    in_channels = 1
    # Get train and test loaders for dataset
    loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader


def train_validate(E, G, D, EG_optim, D_optim, loader, epoch, is_train):

    img_shape = loader.img_shape

    data_loader = loader.train_loader if is_train else loader.test_loader

    E.train() if is_train else E.eval()
    G.train() if is_train else G.eval()
    D.train() if is_train else D.eval()

    vae_batch_loss = 0
    generator_batch_loss = 0
    discriminator_batch_loss = 0

    # reporting all the losses
    vae_batch_loss_recon = 0
    vae_batch_loss_kl = 0
    generator_batch_loss = 0

    # discriminator score on x and x_hat
    score_dx = 0
    score_d_x_hat_1 = 0

    # loss_bce = nn.BCELoss(reduction='mean')
    loss_mse = nn.MSELoss(reduction='sum')

    for batch_idx, (x, _) in enumerate(data_loader):

        batch_size = x.size(0)

        x = x.cuda() if args.cuda else x
        x = x.view(batch_size, -1)

        if is_conv:
            x = x.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        #############################################
        # Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        #
        if is_train:
            D.zero_grad()

        # noise sample
        z_draw = sample_gauss_noise(batch_size, args.latent_size)
        z_draw = z_draw.cuda() if args.cuda else z_draw
        x_gen = G(z_draw)

        if is_conv:
            x_gen = x_gen.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        y_gen = D(x_gen)
        y_real = D(x)

        # Discriminator loss
        y_ones = torch.ones(batch_size, 1)
        y_zeros = torch.zeros(batch_size, 1)

        y_ones = y_ones.cuda() if args.cuda else y_ones
        y_zeros = y_zeros.cuda() if args.cuda else y_zeros

        # Discriminator loss
        d_loss_real = loss_bce(y_real, y_ones)
        d_loss_fake = loss_bce(y_gen, y_zeros)

        discriminator_loss = d_loss_real + d_loss_fake

        if is_train:
            d_loss_real.backward()
            d_loss_fake.backward()
            D_optim.step()

        # logging
        score_dx += y_real.data.mean()
        score_d_x_hat_1 += y_gen.data.mean()
        discriminator_batch_loss += discriminator_loss.item() / batch_size

        #############################################
        # Update Generator/VAE network
        if is_train:
            E.zero_grad()
            G.zero_grad()

        # Encoder - Generator update
        z_x, z_x_mu, z_x_logvar = E(x)

        # Generator forward
        x_hat = G(z_x)

        if is_conv:
            x_hat = x_hat.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        # Loss 1, kl divergence
        loss_kld = loss_kl_gauss(z_x_mu, z_x_logvar)

        # Loss 2, reconstruction loss
        loss_recon = loss_mse(x_hat.view(-1, 1), x.view(-1, 1))

        vae_loss = loss_kld + loss_recon

        if is_train:
            vae_loss.backward(retain_graph=True)
            EG_optim.step()

        vae_batch_loss += vae_loss.item() / batch_size
        vae_batch_loss_recon += loss_recon.item() / batch_size
        vae_batch_loss_kl += loss_kld.item() / batch_size

        #############################################
        # (3) Update G network: maximize log(D(G(z)))

        # Real data forward
        z_x, z_x_mu, z_x_logvar = E(x)

        x_hat = G(z_x)

        if is_conv:
            x_hat = x_hat.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        y_real = D(x_hat)

        generator_loss = loss_bce(y_real, y_ones)

        if is_train:
            generator_loss.backward()
            EG_optim.step()

        generator_batch_loss += generator_loss.item() / batch_size

    print('D(x): %.4f D(G(z)): %.4f' % (score_dx / (batch_idx + 1), score_d_x_hat_1 / (batch_idx + 1)))
    print('Generator loss check: %.4f' % (generator_batch_loss / (batch_idx + 1)))
    print('VAE loss check: recon: %.4f kld: %.4f' % (vae_batch_loss_recon / (batch_idx + 1), vae_batch_loss_kl / (batch_idx + 1)))

    return vae_batch_loss / (batch_idx + 1), generator_batch_loss / (batch_idx + 1), discriminator_batch_loss / (batch_idx + 1)


def execute_graph(E, G, D, EG_optim, D_optim, EG_scheduler, D_scheduler, loader, epoch, use_tb):
    print('=> epoch: {}'.format(epoch))
    # Training loss
    VAE_t_loss, G_t_loss, D_t_loss = train_validate(E, G, D, EG_optim, D_optim, loader, epoch, is_train=True)

    # Validation loss
    VAE_v_loss, G_v_loss, D_v_loss = train_validate(E, G, D, EG_optim, D_optim, loader, epoch, is_train=False)

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
        sample = ibn_generation_example(G, args.latent_size, 8, loader.img_shape, args.cuda)
        logger.add_image('generation example', sample, epoch)

        # Reconstruction example
        sample_x, sample_xhat = ibn_reconstruction_example(E, G, loader.test_loader, 8, loader.img_shape, is_conv, args.cuda)
        logger.add_image('original', sample_x, epoch)
        logger.add_image('reconstruction', sample_xhat, epoch)

    # Manifold example
    if args.latent_size == 2:
        sample = manifold_generation_example(G, loader.img_shape[1:], epoch, is_conv, args.cuda)

        logger.add_image('manifold example', sample, epoch)

    EG_scheduler.step(VAE_v_loss)
    D_scheduler.step(D_v_loss)

    return G_v_loss, D_v_loss


encoder_size = args.encoder_size
decoder_size = args.encoder_size
latent_size = args.latent_size
out_channels = loader.img_shape[0]
in_channels = loader.img_shape[0]
channels_gen = args.out_channels


print(np.prod(loader.img_shape))

E = DCGAN2_Encoder(loader.img_shape, out_channels, encoder_size, latent_size).type(dtype)
h_conv_outsize = E.H_conv_out
print(E)

G = DCGAN2_Generator(h_conv_outsize, out_channels, channels_gen, decoder_size, latent_size).type(dtype)

print(G)

D = DCGAN_Discriminator(in_channels).type(dtype)

print(D)

# Set conv. flag for reshaping images
is_conv = is_conv_model(E)

E.apply(init_wgan_weights)
G.apply(init_wgan_weights)
D.apply(init_wgan_weights)

beta1 = 0.5
beta2 = 0.999

EG_optim = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=args.eg_learning_rate, betas=(beta1, beta2))
D_optim = torch.optim.Adam(D.parameters(), lr=args.d_learning_rate, betas=(beta1, beta2))

# Scheduling
EG_scheduler = ExponentialLR(EG_optim, gamma=args.decay_lr)
D_scheduler = ExponentialLR(D_optim, gamma=args.decay_lr)

# Main training loop
best_g_loss = np.inf

# Main training loop
for epoch in range(1, args.epochs):
    g_v_loss, _ = execute_graph(E, G, D, EG_optim, D_optim, EG_scheduler, D_scheduler, loader, epoch, use_tb)

    if g_v_loss < best_g_loss:
        best_g_loss = g_v_loss
        print('Writing model checkpoint')
        state = {
            'epoch': epoch,
            'E': E.state_dict(),
            'G': G.state_dict(),
            'D': D.state_dict(),
            'EG_optim': EG_optim.state_dict(),
            'D_optim': D_optim.state_dict(),
            'EG_scheduler': EG_scheduler.state_dict(),
            'D_scheduler': D_scheduler.state_dict(),
            'val_loss': g_v_loss
        }
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        file_name = 'models/{}_{}_{}_{}_{:04.4f}.pt'.format(timestamp, args.uid, epoch, latent_size, g_v_loss)

        torch.save(state, file_name)

# TensorboardX logger
logger.close()
