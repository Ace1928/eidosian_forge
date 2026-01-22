import logging
from typing import Union
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
def quantize_block(self, weight):
    if len(weight.shape) != 2:
        raise ValueError(f'Only support 2D matrix, but your input has {len(weight.shape)} dimensions.')
    if weight.shape[0] * weight.shape[1] % self.block_size != 0:
        raise ValueError(f'Weight with shape ({weight.shape[0]} x {weight.shape[1]}) is not dividable by block size {self.block_size}.')
    M, N = weight.shape
    device = weight.device
    weight_flatten = weight.flatten()
    weight_block = weight_flatten.reshape(-1, self.block_size)
    if self.method == 'normal':
        weight_max = weight_block.abs().max(dim=-1)[0]
    elif self.method == 'uniform':
        weight_max = weight_block.mean(dim=-1) + 2.5 * weight_block.std(dim=-1)
    else:
        raise NotImplementedError('Method not supported yet.')
    weight_max = weight_max.unsqueeze(-1)
    weight_divabs = weight_block / weight_max
    weight_divabs = weight_divabs.unsqueeze(-1)
    L_reshaped = self.norm_lookup_table.reshape(1, -1)
    abs_diff = torch.abs(weight_divabs - L_reshaped)
    qweight = torch.argmin(abs_diff, dim=-1)
    qweight = qweight.reshape(-1, 8 // self.num_bits)
    qweight_pack = torch.zeros((M * N // 8 * self.num_bits, 1), dtype=torch.uint8, device=device)
    for i in range(8 // self.num_bits):
        qweight[:, i] = qweight[:, i] << i * self.num_bits
        qweight_pack[:, 0] |= qweight[:, i]
    return (qweight_pack, weight_max, weight.shape)