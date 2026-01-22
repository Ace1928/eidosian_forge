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
def get_pred(x):
    x = up(x)
    x = cm(x)
    return F.softmax(x).data.cpu().numpy()