import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def horizontal_axial_2d_distance(H, W, p=2.0):
    d = local_nd_distance(H, W, p=p, weights=(1, 0))
    return d