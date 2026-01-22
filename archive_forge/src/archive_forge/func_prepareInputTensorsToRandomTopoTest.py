import torch
import numpy as np
import argparse
from typing import Dict
def prepareInputTensorsToRandomTopoTest(seed, max_tensor_num, max_tensor_dim, max_tensor_size, debug_tensor, device, dtype):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(0, seed))
    seed_tensor = torch.tensor(np.random.randint(0, seed))
    num_tensor = np.random.randint(1, max_tensor_num)
    tensor_dim = np.random.randint(1, max_tensor_dim)
    tensor_shape = []
    numel = 1
    if debug_tensor:
        tensor_shape.append(1)
    else:
        for i in range(tensor_dim):
            size_i = np.random.randint(1, int(max_tensor_size / numel / 2 ** (tensor_dim - i)))
            size_i = min(size_i, 128 + size_i % 128)
            tensor_shape.insert(0, size_i)
            numel *= size_i
    if DEBUG_PRINT:
        print('output tensor shape: ', tensor_shape)
    num_broadcasted_tensors = np.random.randint(0, 1)
    broadcasted_tensors_indices = np.random.choice(torch.arange(num_tensor), num_broadcasted_tensors, replace=False)
    tensor_list = []
    for i in range(num_tensor):
        if i in broadcasted_tensors_indices:
            compatible_shape = get_broadcast_compatible_shape(tensor_shape)
            tensor_list.append(torch.randn(compatible_shape, device=device, dtype=dtype) * 100)
        else:
            tensor_list.append(torch.randn(tensor_shape, device=device, dtype=dtype) * 100)
    return (seed_tensor, tensor_list)