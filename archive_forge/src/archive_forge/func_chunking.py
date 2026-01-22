import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
def chunking(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    out, rest = self.pad_chunk(x)
    batch_size, feat_dim, seq_len = out.shape
    segments1 = out[:, :, :-self.chunk_stride].contiguous().view(batch_size, feat_dim, -1, self.chunk_size)
    segments2 = out[:, :, self.chunk_stride:].contiguous().view(batch_size, feat_dim, -1, self.chunk_size)
    out = torch.cat([segments1, segments2], dim=3)
    out = out.view(batch_size, feat_dim, -1, self.chunk_size).transpose(2, 3).contiguous()
    return (out, rest)