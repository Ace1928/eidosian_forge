from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def assert_dtypes_valid(lhs_dtype, rhs_dtype, target):
    if not _is_cuda(target):
        assert lhs_dtype == rhs_dtype, f'First input ({lhs_dtype}) and second input ({rhs_dtype}) must have the same dtype!'
        return
    if target.capability < 90:
        assert not lhs_dtype.is_fp8e4nv() and (not rhs_dtype.is_fp8e4nv()), 'Dot op does not support fp8e4nv on CUDA arch < 90'
        if lhs_dtype.is_fp8() and rhs_dtype.is_fp8():
            return
        assert lhs_dtype == rhs_dtype, f'First input ({lhs_dtype}) and second input ({rhs_dtype}) must have the same dtype!'
    else:
        assert not lhs_dtype.is_fp8e4b15() and (not rhs_dtype.is_fp8e4b15()), 'Dot op does not support fp8e4b15 on CUDA arch >= 90'
        assert not lhs_dtype.is_fp8e4b15x4() and (not rhs_dtype.is_fp8e4b15x4()), 'Dot op does not support fp8e4b15x4 on CUDA arch >= 90'
        if lhs_dtype.is_int() or rhs_dtype.is_int():
            assert lhs_dtype == rhs_dtype, f'Both operands must be same type. First operand ({lhs_dtype}) and second operand ({rhs_dtype})'
            assert lhs_dtype.is_int8() or lhs_dtype.is_uint8(), f'Both operands must be either int8 or uint8. Operand type ({lhs_dtype})'
        elif lhs_dtype.is_fp8() or rhs_dtype.is_fp8():
            assert lhs_dtype.is_fp8e4nv() or lhs_dtype.is_fp8e5(), f'Only supports fp8e4nv or fp8e5. First operand ({lhs_dtype})'
            assert rhs_dtype.is_fp8e4nv() or rhs_dtype.is_fp8e5(), f'Only supports fp8e4nv or fp8e5. Second operand ({rhs_dtype})'
        else:
            assert lhs_dtype.is_fp16() or lhs_dtype.is_bf16() or lhs_dtype.is_fp32() or lhs_dtype.is_int1(), f'Unsupported dtype {lhs_dtype}'
            assert rhs_dtype.is_fp16() or rhs_dtype.is_bf16() or rhs_dtype.is_fp32() or rhs_dtype.is_int1(), f'Unsupported dtype {rhs_dtype}'
            assert lhs_dtype == rhs_dtype, f'First input ({lhs_dtype}) and second input ({rhs_dtype}) must have the same dtype!'