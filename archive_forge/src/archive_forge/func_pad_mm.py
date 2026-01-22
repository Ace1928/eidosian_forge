import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def pad_mm(mat1: Tensor, mat2: Tensor, m_padded_length: int, k_padded_length: int, n_padded_length: int) -> Tensor:
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 1)
        mat2 = pad_dim(mat2, k_padded_length, 0)
        return torch.ops.aten.mm(mat1, mat2)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 1)
        return torch.ops.aten.mm(mat1, mat2)[:, :-n_padded_length]
    else:
        mat1 = pad_dim(mat1, m_padded_length, 0)
        return torch.ops.aten.mm(mat1, mat2)[:-m_padded_length, :]