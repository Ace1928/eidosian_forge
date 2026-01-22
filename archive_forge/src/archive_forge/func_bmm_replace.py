import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
    if m_padded_length != 0 or k_padded_length != 0 or n_padded_length != 0:
        return pad_bmm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)
    return aten.bmm(mat1, mat2)