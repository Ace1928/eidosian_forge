import logging
import math
from dataclasses import dataclass
import torch
from xformers import _is_triton_available
from xformers.components.attention import Attention, AttentionConfig, register_attention
def create_triton_kernels(self, device):
    self.sparse_dot_sdd = blocksparse_matmul(self.layout, self.block_size, 'sdd', trans_a=False, trans_b=True, device=device)
    self.sparse_dot_dsd = blocksparse_matmul(self.layout, self.block_size, 'dsd', trans_a=False, trans_b=False, device=device)
    self.sparse_softmax = blocksparse_softmax(self.layout, self.block_size, device=device)