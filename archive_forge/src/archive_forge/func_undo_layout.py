from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
def undo_layout(permuted_tensor: torch.Tensor, tile_indices: torch.LongTensor) -> torch.Tensor:
    """
    Undo a tiled permutation such as turing or ampere layout

    :param permuted_tensor: torch tensor in a permuted layout
    :param tile_indices: reverse transformation indices, from get_inverse_transform_indices
    :return: contiguous row-major tensor
    """
    (rows, cols), (tile_rows, tile_cols) = (permuted_tensor.shape, tile_indices.shape)
    assert rows % tile_rows == cols % tile_cols == 0, 'tensor must contain a whole number of tiles'
    tensor = permuted_tensor.reshape(-1, tile_indices.numel()).t()
    outputs = torch.empty_like(tensor)
    outputs[tile_indices.flatten()] = tensor
    outputs = outputs.reshape(tile_rows, tile_cols, cols // tile_cols, rows // tile_rows)
    outputs = outputs.permute(3, 0, 2, 1)
    return outputs.reshape(rows, cols).contiguous()