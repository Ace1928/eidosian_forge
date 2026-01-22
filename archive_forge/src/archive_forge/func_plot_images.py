import ray
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def plot_images(dataloader):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Original Images')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()