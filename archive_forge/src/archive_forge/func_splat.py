from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def splat(value: tl.tensor, shape: List[int], builder: ir.builder) -> tl.tensor:
    assert not value.type.is_block(), 'Cannot splat a block tensor'
    if len(shape) == 0:
        return value
    ret_ty = tl.block_type(value.dtype, shape)
    return tl.tensor(builder.create_splat(value.handle, shape), ret_ty)