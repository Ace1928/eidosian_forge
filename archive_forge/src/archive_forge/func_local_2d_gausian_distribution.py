import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def local_2d_gausian_distribution(H, W, sigma=1):
    return local_nd_gaussian_distribution(H, W, sigma=sigma)