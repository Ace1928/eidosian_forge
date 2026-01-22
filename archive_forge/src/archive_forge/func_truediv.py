from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def truediv(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
        other = cast(other, input_scalar_ty, builder)
    elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
        input = cast(input, other_scalar_ty, builder)
    elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
        input = cast(input, tl.float32, builder)
        other = cast(other, tl.float32, builder)
    elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
        if input_scalar_ty.fp_mantissa_width > other_scalar_ty.fp_mantissa_width:
            other = cast(other, input_scalar_ty, builder)
        else:
            input = cast(input, other_scalar_ty, builder)
    else:
        assert False
    return tl.tensor(builder.create_fdiv(input.handle, other.handle), input.type)