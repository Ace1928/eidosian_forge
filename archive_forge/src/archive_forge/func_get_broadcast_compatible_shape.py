import torch
import numpy as np
import argparse
from typing import Dict
def get_broadcast_compatible_shape(tensor_shape):
    max_dim = len(tensor_shape)
    num_b_dims = np.random.randint(0, max_dim + 1)
    trim_head = np.random.randint(0, min(num_b_dims + 1, max_dim))
    shape = tensor_shape[trim_head:max_dim]
    for i in np.random.choice(range(max_dim - trim_head), num_b_dims - trim_head, replace=False):
        shape[i] = 1
    return shape