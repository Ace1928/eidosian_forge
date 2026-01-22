import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def pad_addmm(input: Optional[Tensor], mat1: Tensor, mat2: Tensor, m_padded_length: int, k_padded_length: int, n_padded_length: int, beta=1.0, alpha=1.0):
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 1)
        mat2 = pad_dim(mat2, k_padded_length, 0)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 1)
    elif m_padded_length != 0:
        mat1 = pad_dim(mat1, m_padded_length, 0)
    if input is not None and k_padded_length == 0:
        if n_padded_length != 0:
            if input.dim() == 2 and input.shape[1] != 1:
                input = pad_dim(input, n_padded_length, 1)
            elif input.dim() == 1 and input.shape[0] != 1:
                input = pad_dim(input, n_padded_length, 0)
        elif m_padded_length != 0 and input.dim() == 2 and (input.shape[0] != 1):
            input = pad_dim(input, m_padded_length, 0)
    if k_padded_length != 0:
        return addmm_replace(input, mat1, mat2, beta=beta, alpha=alpha)
    elif n_padded_length != 0:
        return addmm_replace(input, mat1, mat2, beta=beta, alpha=alpha)[:, :-n_padded_length]
    else:
        return addmm_replace(input, mat1, mat2, beta=beta, alpha=alpha)[:-m_padded_length, :]