import gc
import math
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import pretty_barplot
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import SparseCS, _matmul_with_mask
def densify_mask(mask, config):
    num_heads = config.num_heads
    seq_length = config.seq_length
    block_size = config.block_size
    dense_mask = torch.zeros(num_heads, seq_length, seq_length)
    for h, i, j in zip(*mask.nonzero(as_tuple=True)):
        dense_mask[h, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = mask[h, i, j]
    return dense_mask