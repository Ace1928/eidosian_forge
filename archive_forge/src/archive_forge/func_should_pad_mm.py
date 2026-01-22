import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def should_pad_mm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ('mat1', 'mat2'))
    return should_pad_common(mat1, mat2) and should_pad_bench(mat1, mat2, torch.ops.aten.mm)