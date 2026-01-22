import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
def check_lastdim_alignment_stride1(reasons: List[str], name: str, x: torch.Tensor, alignment: int) -> None:
    if x.shape[-1] % alignment != 0:
        reasons.append(f'{name}.shape[-1] % {alignment} != 0')
    elif x.stride(-2) % alignment != 0:
        reasons.append(f'{name}.stride(-2) % {alignment} != 0 ({name}.stride() = {x.stride()})')
    if x.stride(-1) > 1:
        reasons.append(f'{name}.stride(-1) > 1 ({name}.stride() = {x.stride()}) - you should call `.contiguous()` on the input')