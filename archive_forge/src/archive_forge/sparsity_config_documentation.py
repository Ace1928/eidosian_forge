import random
import torch
Generates edited `Longformer` sparsity layout used by each head in the sparse attention.
        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing `BSLongformer`
                sparsity layout of all head
        