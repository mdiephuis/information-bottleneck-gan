import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import os
import torchvision.utils as tvu
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Variable containing the loss.
    """

    N, _ = scores_real.size()
    dtype = scores_fake.type()
    id_mat = torch.ones(N, ).type(dtype)
    loss_real = 0.5 * torch.mean(torch.pow(scores_real - id_mat, 2))
    loss_fake = 0.5 * torch.mean(torch.pow(scores_fake, 2))

    return loss_real + loss_fake


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Variable containing the loss.
    """
    N, _ = scores_fake.size()
    dtype = scores_fake.type()
    id_mat = torch.ones(N, ).type(dtype)

    loss = 0.5 * torch.mean(torch.pow(scores_fake - id_mat, 2))
    return loss


def loss_bce(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
def loss_sigmoid_cross_entropy_with_logits(x_hat, x):
    loss = x_hat.clamp(min=0) - x_hat * x + torch.log(1 + torch.exp(-torch.abs(x_hat)))
    return torch.mean(loss)


def loss_kl_gauss(mu, log_var):
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD


def sample_uniform_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)


def sample_gauss_noise(batch_size, dim, mu=0, std=1):
    return torch.Tensor(batch_size, dim).normal_(mu, std)


def init_xavier_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_xavier_weights(sub_mod)


def init_normal_weights(module, mu, std):
    for m in module.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(mu, std)
            m.bias.data.zero_()
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_normal_weights(sub_mod, mu, std)


def init_wgan_weights(m):
    # https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def is_conv_model(model):
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            return True
    return False


def im2tensorboard(sample):

    if len(sample.shape) == 2:
            sample = sample = sample[:, :, np.newaxis]

    sample = np.transpose(sample, (2, 0, 1))
    return torch.from_numpy(sample)


def ibn_generation_example(G, noise_dim, n_samples, img_shape, use_cuda):

    z_real = sample_gauss_noise(n_samples, noise_dim)
    z_real = z_real.cuda() if use_cuda else z_real

    x_hat = G(z_real).cpu().detach()
    x_hat = x_hat * 0.5 + 0.5
    x_hat = torch.transpose(x_hat, 1, 2)
    x_hat = torch.transpose(x_hat, 2, 3)

    x_hat = x_hat.reshape(n_samples * img_shape[1], img_shape[2], img_shape[0])

    sample = tvu.make_grid(x_hat, normalize=True, scale_each=True)
    if sample.size(-1) == 1:
        sample = sample.squeeze(-1)

    return sample.numpy()


def vaegan_generation_example(G, noise_dim, n_samples, img_shape, use_cuda):

    z_real = sample_gauss_noise(n_samples, noise_dim)
    z_real = z_real.cuda() if use_cuda else z_real
    x_hat = G(z_real)
    x_hat = x_hat.cpu().view(n_samples, img_shape[0], img_shape[1], img_shape[2])

    # due to tanh output layer in the generator
    x_hat = x_hat * 0.5 + 0.5

    return x_hat


def ibn_reconstruction_example(E, G, test_loader, n_samples, img_shape, is_conv, use_cuda):
    E.eval()
    G.eval()

    x, _ = next(iter(test_loader))

    n_samples = min((n_samples, x.size(0)))

    if is_conv:
        x = x.view(-1, img_shape[0], img_shape[1], img_shape[2])
    else:
        x = x.view(x.size(0), -1)

    x = x.cuda() if use_cuda else x

    z_val, _, _ = E(x)

    x_hat = G(z_val)

    x_hat = x_hat * 0.5 + 0.5

    if len(x.size()) == 3:
        x = x.unsqueeze(1)
        x_hat = x_hat.unsqueeze(1)

    x = x[:n_samples].cpu()
    x = torch.transpose(x[:n_samples].cpu(), 1, 2)
    x = torch.transpose(x[:n_samples].cpu(), 2, 3)

    x = x.reshape(n_samples * img_shape[1], img_shape[2], img_shape[0])

    x_hat = x_hat[:n_samples].cpu()
    x_hat = torch.transpose(x_hat, 1, 2)
    x_hat = torch.transpose(x_hat, 2, 3)
    x_hat = x_hat.reshape(n_samples * img_shape[1], img_shape[2], img_shape[0])

    comparison = torch.cat((x, x_hat), 1)

    reconstructed = tvu.make_grid(comparison.detach().cpu(), normalize=True, scale_each=True)
    return reconstructed.numpy()


def manifold_generation_example(G, img_shape, epoch, use_cuda):
    z_range = 1
    nx, ny = 15, 15

    z1 = np.linspace(- z_range, z_range, ny)
    z2 = np.linspace(- z_range, z_range, nx)
    manifold = np.zeros(shape=(img_shape[0] * nx, img_shape[1] * ny))
    x_pixel, y_pixel = 0, 0
    for i in z1:
        for j in z2:
            z = torch.FloatTensor([i, j])
            z = z.cuda() if use_cuda else z
            sample = G(z).cpu().detach().numpy().reshape(img_shape[0], img_shape[1])
            manifold[x_pixel:x_pixel + img_shape[0], y_pixel:y_pixel + img_shape[1]] = sample
            y_pixel += img_shape[1]
        x_pixel += img_shape[0]
        y_pixel = 0
    plt.imshow(manifold, extent=[- z_range, z_range, - z_range, z_range])
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/manifold_example_{}.png'.format(epoch))
    return torch.from_numpy(manifold)
