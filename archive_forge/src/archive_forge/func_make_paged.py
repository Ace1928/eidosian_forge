import math
from dataclasses import dataclass
from typing import (
import torch
def make_paged(self, block_tables: torch.Tensor, page_size: int, paged_type: Type['PagedBlockDiagonalPaddedKeysMask']) -> AttentionBias:
    paged_bias = paged_type(q_seqinfo=self.q_seqinfo, k_seqinfo=self.k_seqinfo, block_tables=block_tables, page_size=page_size)
    paged_bias.k_seqinfo.padding = block_tables.shape[1] * page_size
    return paged_bias