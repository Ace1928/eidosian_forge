import torch
import torch.nn.functional as F
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import standard_kwargs, forward_helper, set_grad_sample_if_exists
from typing import List, Optional
def weight_per_sample_grad(weight):
    batch_size = input.shape[0]
    embedding_dim = weight.shape[1]
    index = input.unsqueeze(-1).expand(*input.shape, embedding_dim).reshape(batch_size, -1, embedding_dim)
    grad_sample = torch.zeros(batch_size, *weight.shape, device=weight.device, dtype=grad_output.dtype)
    return grad_sample.scatter_add_(1, index, grad_output.reshape(batch_size, -1, embedding_dim))