from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def shl(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_shl(input.handle, other.handle), input.type)