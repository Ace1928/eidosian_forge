import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def should_pad_common(mat1: Tensor, mat2: Tensor, input: Optional[Tensor]=None) -> bool:
    return torch._inductor.config.shape_padding and check_device(mat1, mat2) and check_dtype(mat1, mat2) and (not any_is_symbolic(mat1, mat2, input))