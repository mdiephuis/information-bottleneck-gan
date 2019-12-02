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
parser.add_argument('--learning-rate', type=float, default=0.0002,
                    help='Learning rate (default: 1e-4')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')

parser.add_argument('--ngf', type=int, default=64)

parser.add_argument('--ndf', type=int, default=64)

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
transforms = [transforms.Scale(64), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
# transforms = None

# Get train and test loaders for dataset
loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


def train_validate(G, D, GE_optim, D_optim, loader, epoch, is_train):

    img_shape = loader.img_shape

    data_loader = loader.train_loader if is_train else loader.test_loader

    G.train() if is_train else G.eval()
    D.train() if is_train else D.eval()

    VAE_batch_loss = 0
    G_batch_loss = 0
    D_batch_loss = 0

    # discriminator score on x and x_hat
    score_dx = 0
    score_d_x_hat_1 = 0
    score_d_x_hat_2 = 0

    loss_bce_sum = nn.BCELoss()
    loss_mse = nn.MSELoss()

    for batch_idx, (x, _) in enumerate(data_loader):

        batch_size = x.size(0)

        x = x.cuda() if args.cuda else x
        x = x.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        eta = sample_gauss_noise(batch_size, img_shape[1] * img_shape[2], 0, 0.1)

        eta = eta.cuda() if args.cuda else eta

        #x += eta.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        # Discriminator loss
        y_ones = torch.ones(batch_size, 1)
        y_zeros = torch.zeros(batch_size, 1)

        y_ones = y_ones.cuda() if args.cuda else y_ones
        y_zeros = y_zeros.cuda() if args.cuda else y_zeros

        # Real data, discriminator forward
        if is_train:
            D.zero_grad()
        y_real = D(x)
        errD_real = loss_bce_sum(y_real, y_ones)
        errD_real.backward()

        # Encoder forward
        fixed_noise = torch.FloatTensor(batch_size, args.latent_size, 1, 1).normal_(0, 1)
        fixed_noise = fixed_noise.cuda() if args.cuda else fixed_noise

        # Generator forward
        x_hat = G.decoder(fixed_noise)
        y_hat = D(x_hat.view(batch_size, img_shape[0], img_shape[1], img_shape[2]))
        errD_fake = loss_bce_sum(y_hat, y_zeros)
        errD_fake.backward()
        #
        score_dx += y_real.data.mean()
        score_d_x_hat_1 += y_hat.data.mean()

        # Discriminator loss
        discriminator_loss = errD_real + errD_fake

        D_batch_loss += discriminator_loss.item() / batch_size

        if is_train:
            D_optim.step()
        # RRound 2
        # Encoder forward
        if is_train:
            G.zero_grad()
        z_hat, z_mu, z_logvar = G.encoder(x)

        # Generator forward
        x_hat = G.decoder(z_hat)
        y_hat = D(x_hat.view(batch_size, img_shape[0], img_shape[1], img_shape[2]))

        #
        score_d_x_hat_2 += y_hat.data.mean()

        loss_recon = loss_mse(x_hat.view(-1, 1), x.view(-1, 1))
        # Loss 1, kl divergence
        loss_kld = loss_kl_gauss(z_mu, z_logvar)

        VAE_loss = loss_kld + loss_recon

        VAE_batch_loss += VAE_loss.item() / batch_size

        if is_train:
            VAE_loss.backward()
            GE_optim.step()

        # Discriminator loss
        y_ones = torch.ones(batch_size, 1)
        y_ones = y_ones.cuda() if args.cuda else y_ones

        # Discriminator loss
        x_hat = G(x)
        y_real = D(x_hat)
        generator_loss = loss_bce_sum(y_real, y_ones)

        G_batch_loss += generator_loss.item() / batch_size

        if is_train:
            generator_loss.backward()
            GE_optim.step()

    print('D(x): %.4f D(G(z)): %.4f , %.4f' % (score_dx / (batch_idx + 1), score_d_x_hat_1 / (batch_idx + 1), score_d_x_hat_2 / (batch_idx + 1)))

    return VAE_batch_loss / (batch_idx + 1), G_batch_loss / (batch_idx + 1), D_batch_loss / (batch_idx + 1)


def execute_graph(G, D, GE_optim, D_optim, loader, epoch, use_tb):
    print('=> epoch: {}'.format(epoch))
    # Training loss
    VAE_t_loss, G_t_loss, D_t_loss = train_validate(G, D, GE_optim, D_optim, loader, epoch, is_train=True)

    # Validation loss
    VAE_v_loss, G_v_loss, D_v_loss = train_validate(G, D, GE_optim, D_optim, loader, epoch, is_train=False)

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
        sample = generation_example(G, args.latent_size, 10, img_shape, args.cuda)
        sample = sample.detach()
        sample = tvu.make_grid(sample, normalize=True, scale_each=True)
        logger.add_image('generation example', sample, epoch)

    # Reconstruction example
        reconstructed = reconstruction_example(G, loader.test_loader, 10, img_shape, args.cuda)
        reconstructed = reconstructed.detach()
        reconstructed = tvu.make_grid(reconstructed, normalize=True, scale_each=True)
        logger.add_image('reconstruction example', reconstructed, epoch)

    return G_v_loss, D_v_loss


# MNIST Model definitions
nz = args.nz
ngf = args.ngf
ndf = args.ndf
nc = 1
imageSize = 64

# E = Encoder(imageSize, ngf, nc, nz).type(dtype)
G = Generator(imageSize, ngf, nc, nz).type(dtype)
print(G)
D = Discriminator(imageSize, ngf, ndf, nc).type(dtype)
print(D)


# E.apply(init_xavier_weights)
G.apply(init_normal_weights)
D.apply(init_normal_weights)


beta1 = 0.5
beta2 = 0.999

# E_optim = torch.optim.RMSprop(E.parameters(), lr=1e-3, weight_decay=1e-5)
GE_optim = torch.optim.Adam(G.parameters(), lr=args.learning_rate, betas=(beta1, beta2))
D_optim = torch.optim.Adam(D.parameters(), lr=args.learning_rate, betas=(beta1, beta2))


# Main training loop
for epoch in range(1, args.epochs):
    _, _ = execute_graph(G, D, GE_optim, D_optim, loader, epoch, use_tb)


# TensorboardX logger
logger.close()
