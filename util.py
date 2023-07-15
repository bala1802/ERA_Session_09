import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_tensor

import numpy as np

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)


class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def load_data(mode, transform):
   return Cifar10SearchDataset(root='./data', train=(mode=="train"),
                                        download=True, transform=transform)

def get_transforms(mode):
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.2470, 0.2435, 0.2616]

    if mode == 'train':
        return A.Compose([
                            A.Normalize(mean=means, std=stds, always_apply=True),
                            A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                            A.RandomCrop(height=32, width=32, always_apply=True),
                            A.HorizontalFlip(),
                            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                            ToTensorV2(),
                        ])
    else:
        return A.Compose([
                            A.Normalize(mean=means, std=stds, always_apply=True),
                            ToTensorV2(),
                        ])

def construct_loader(data):
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    return torch.utils.data.DataLoader(data, **dataloader_args)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_classes():
    return ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse','ship', 'truck')

def display_data(data_loader):
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:4]))
    classes = load_classes()
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

def load_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    