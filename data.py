from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import cv2


class Loader(object):
    def __init__(self, dataset_ident, file_path, download, batch_size, data_transform, target_transform, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        # This seems to be a bug, but disabling pin memory is way faster when using the GPU
        #kwargs = {}

        # set the dataset
        # NOTE: will need a refractor one we load more different datasets, that require custom classes
        loader_map = {
            'mnist': datasets.MNIST,
            'MNIST': datasets.MNIST,
            'FashionMNIST': datasets.FashionMNIST,
            'fashion': datasets.FashionMNIST
        }

        num_class = {
            'mnist': 10,
            'MNIST': 10,
            'fashion': 10,
            'FashionMNIST': 10
        }

        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(loader_map[dataset_ident], file_path, download,
                                                       data_transform, target_transform)
        # Set the loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        # infer and set size, idea from:
        # https://github.com/jramapuram/helpers/
        tmp_batch, _ = self.train_loader.__iter__().__next__()
        self.img_shape = list(tmp_batch.size())[1:]
        self.num_class = num_class[dataset_ident]
        # Note, might be incorrect for last batch in an iteration
        self.batch_size = batch_size

    @staticmethod
    def get_dataset(dataset, file_path, download, data_transform, target_transform):

        # Check for transform to be None, a single item, or a list
        # None -> default to transform_list = [transforms.ToTensor()]
        # single item -> list
        if not data_transform:
            data_transform = [transforms.ToTensor()]
        elif not isinstance(data_transform, list):
            data_transform = list(data_transform)

        # Training and Validation datasets
        train_dataset = dataset(file_path, train=True, download=download,
                                transform=transforms.Compose(data_transform),
                                target_transform=target_transform)

        test_dataset = dataset(file_path, train=False, download=download,
                               transform=transforms.Compose(data_transform),
                               target_transform=target_transform)

        return train_dataset, test_dataset


class gauss_circle(object):
    def __init__(self, size=2, num_circles=8, std=0.02, radius=2, uniform=False):
        self.num_circles = num_circles
        self.std = std
        self.size = size
        self.radius = radius
        delta = 2 * np.pi / self.num_circles

        centers_x = np.asarray([self.radius * np.cos(i * delta) for i in range(self.num_circles)])
        centers_y = np.asarray([self.radius * np.sin(i * delta) for i in range(self.num_circles)])

        if self.uniform is True:
            # Strictly Uniform
            self.p = [1. / self.num_circles for _ in range(self.num_circles)]
        else:
            # Random draw from uniform distribution
            self.p = [np.random.uniform() for _ in range(self.num_circles)]
            self.p /= np.sum(self.p)

        self.centers = np.concatenate((centers_x[:, np.newaxis], centers_y[:, np.newaxis]), 1)

    def sample(self, n_samples):
        centers_idx = np.random.choice(self.num_circles, n_samples, p=self.p)
        centers_sample = self.centers[centers_idx, :]
        data_sample = np.random.normal(loc=centers_sample, scale=self.std)
        return data_sample.astype('float32')


class CELEBA(Dataset):
    """
    loader for the CELEB-A dataset
    """
    def __init__(self, data_folder):

        self.len = len(os.listdir(data_folder))
        self.data_names = [os.path.join(data_folder, name) for name in sorted(os.listdir(data_folder))]

        self.len = len(self.data_names)

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, item):

        image = cv2.cvtColor(cv2.imread(self.data_names[item]), cv2.COLOR_BGR2RGB)

        # CHANNEL FIRST
        image = image.transpose(2, 0, 1)
        image = image.astype("float32")

        return image
