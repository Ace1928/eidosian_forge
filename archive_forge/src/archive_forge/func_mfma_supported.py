from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def mfma_supported(M, N, K, allow_tf32, ret_scalar_ty) -> bool:
    if not gpu_has_mfma():
        return False
    return True