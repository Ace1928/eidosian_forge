from dataclasses import dataclass
import torch
from xformers.components.attention import Attention, AttentionConfig, register_attention
def calc_rel_pos(n: int):
    rel_pos = torch.arange(n)[None, :] - torch.arange(n)[:, None]
    rel_pos += n - 1
    return rel_pos