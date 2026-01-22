import torch
import torch.distributed as dist
from torch import nn
def dequantize_and_aggregate(fut):
    all_ranks_quantized_tensor = fut.wait()[0]
    aggregated_dequantized_tensor = torch.zeros_like(all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32)
    for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
        aggregated_dequantized_tensor += _dequantize_per_channel_cuda(quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1])
    return torch.flatten(aggregated_dequantized_tensor).cuda(tensor.device)[:tensor.size()[0]] / world_size