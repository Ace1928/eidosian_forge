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
def gen_window_mask(self, seq_length):
    num_window_blocks = self.config.num_window_tokens // self.config.block_size
    if num_window_blocks % 2 == 0:
        num_window_blocks += 1
    return local_1d_pattern(seq_length, num_window_blocks)