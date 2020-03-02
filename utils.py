import torch
import torch.nn as nn


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


def normal_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dcgan_generation_example(G, noise_dim, n_samples, img_shape, use_cuda):

    z_real = sample_uniform_noise(n_samples, noise_dim)
    z_real = z_real.cuda() if use_cuda else z_real

    x_hat = G(z_real).cpu().view(n_samples, img_shape[0], img_shape[1], img_shape[2])

    # due to tanh output layer in the generator
    #x_hat = x_hat * 0.5 + 0.5

    return x_hat


def mnist_generation_example(G, noise_dim, n_samples, img_shape, use_cuda):

    z_real = sample_gauss_noise(n_samples, noise_dim)
    z_real = z_real.cuda() if use_cuda else z_real

    x_hat = G.decoder(z_real).cpu().view(n_samples, img_shape[0], img_shape[1], img_shape[2])

    return x_hat


def dcgan_reconstruction_example(E, G, test_loader, n_samples, img_shape, use_cuda):
    E.eval()
    G.eval()

    x, _ = next(iter(test_loader))

    x = x.view(x.size(0), -1)

    #x = x * 0.5 + 0.5
    x = x.cuda() if use_cuda else x

    # x = x.view(-1, img_shape[0], img_shape[1], img_shape[2])

    z_val, _, _ = E(x)

    x_hat = G(z_val)
    #x_hat = x_hat * 0.5 + 0.5

    x = x[:n_samples].cpu().view(10 * img_shape[1], img_shape[2])
    x_hat = x_hat[:n_samples].cpu().view(10 * img_shape[1], img_shape[2])
    comparison = torch.cat((x, x_hat), 1).view(10 * img_shape[1], 2 * img_shape[2])

    return comparison
