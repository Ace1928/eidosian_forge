from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def elementwise_dtypes(*_args, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND) -> Tuple[torch.dtype, torch.dtype]:
    """
    Computes the computation and result dtypes for elementwise type promotion
    on the given arguments and with the given elementwise type promotion kind.

    Note that not all inputs to an elementwise operation necessarily participate in type promotion.
    For example, the "alpha" parameter of torch.add does not participate in type promotion,
    although it may be cast to the Python type corresponding to the computation dtype that
    the type promotion algorithm determines.

    Default elementwise type promotion, which all other type promotion kinds tweak (see below),
    first decides which of four ordered types to use:

    bool -> integer -> floating point -> complex

    The selected type is the "lowest" type in the above list such that all number arguments
    have a weakly "lower" type and all tensor arguments have a weakly lower corresponding
    type for their dtype.

    Once the type is determined, the particular result dtype is found. The dtypes are
    partially ordered as follows:

    bool -> uint8, int8 -> int16 -> int32 -> int64 ->
      float16, bfloat16 -> float32 -> float64 -> complex32 -> complex64 -> complex128

    The result dtype is selected by:
      - if no tensor's dtype has the same corresponding type as the one selected,
          then the result dtype is the (default) dtype corresponding to the selected type
          (for example, 1.5 + an integer tensor has a result dtype of the default floating point dtype)
      - if the result type is complex then the dtype is:
        -  the default complex dtype if there are no floating point or complex tensors
        -  if there are floating point or complex tensors with one or more dimensions, then
            the complex dtype corresponding to the highest corresponding complex dtype among those tensors
            (for example, double + cfloat -> cdouble)
        -  if there are only floating point or complex tensors with zero dimensions, then
            the complex dtype corresponding to the highest corresponding complex dtype among those tensors
      - if the first two cases do not apply, the result dtype is the highest dtype among
          all tensors with one or more dimensions of the output type, and if there are no such
          tensors then it's the highest dtype among all tensors with zero dimensions of the output type
          (for example, long + half -> half, even if the half tensor has zero dimensions)

    The "corresponding complex dtypes" are:
      float16    -> complex32
      bfloat16   -> complex64
      float32    -> complex64
      float64    -> complex128
      complex32  -> complex32
      complex64  -> complex64
      complex128 -> complex128

    The DEFAULT type promotion kind computes per above, and then uses the result dtype to pick a computation
    dtype by mapping low precision floating point and complex dtypes as follows:

      float16   -> float32
      bfloat16  -> float32
      complex32 -> complex64

    This is referred to as "op math", and the NO_OPMATH type promotion kind disables this mapping, making the
    computation dtype the same as the result dtype when it's selected. NO_OPMATH is appropriate for kernels
    which perform no mathematical operations on their tensors (see below for examples).

    The INT_TO_FLOAT type promotion kind maps boolean and integer result dtypes to the default floating point dtype,
    and computation dtypes to the appropriate op math dtype.

    The COMPLEX_TO_FLOAT type promotion kind maps complex result dtypes to the corresponding float dtype, following this
    mapping:

        complex32  -> float16
        complex64  -> float32
        complex128 -> float64

    Note that COMPLEX_TO_FLOAT derives the computation dtype as the DEFAULT setting does.

    The BOOL_TO_LONG type promotion kind maps boolean computation and result dtypes to long.

    The ALWAYS_BOOL type promotion kind always sets the result dtype to bool.

    Example operators for each type promotion option:
      DEFAULT                 : add
      NO_OPMATH               : where, nextafter, cat
      INT_TO_FLOAT            : sin
      COMPLEX_TO_FLOAT        : abs
      BOOL_TO_LONG            : pow
      ALWAYS_BOOL             : eq

    """
    args = tuple((x for x in _args if x is not None))
    highest_type: type = bool
    import sympy
    for x in args:
        if not isinstance(x, (Number, TensorLike, sympy.Symbol)):
            msg = f'Unexpected type {str(type(x))} when computing elementwise type promotion!'
            raise ValueError(msg)
        if isinstance(x, Number):
            highest_type = get_higher_type(highest_type, number_type(x))
        elif isinstance(x, sympy.Symbol):
            highest_type = get_higher_type(highest_type, symbol_type(x))
        else:
            highest_type = get_higher_type(highest_type, dtype_to_type(x.dtype))
    result_dtype = None

    def _find_highest_dtype_filtered(args, filter, *, float_as_complex=False) -> Optional[torch.dtype]:
        zero_dim_tensor_dtype = None
        one_plus_dim_tensor_dtype = None
        for x in args:
            if isinstance(x, TensorLike) and filter(x.dtype):
                _dtype = x.dtype
                if float_as_complex and is_float_dtype(_dtype):
                    _dtype = corresponding_complex_dtype(_dtype)
                if x.ndim == 0:
                    zero_dim_tensor_dtype = get_higher_dtype(zero_dim_tensor_dtype, _dtype)
                else:
                    one_plus_dim_tensor_dtype = get_higher_dtype(one_plus_dim_tensor_dtype, _dtype)
        if one_plus_dim_tensor_dtype is not None:
            return one_plus_dim_tensor_dtype
        return zero_dim_tensor_dtype
    if highest_type is float:
        result_dtype = _find_highest_dtype_filtered(args, is_float_dtype)
        result_dtype = torch.get_default_dtype() if result_dtype is None else result_dtype
    elif highest_type is complex:
        result_dtype = _find_highest_dtype_filtered(args, lambda x: is_float_dtype(x) or is_complex_dtype(x), float_as_complex=True)
        if result_dtype is None:
            result_dtype = corresponding_complex_dtype(torch.get_default_dtype())
    elif highest_type is int:
        result_dtype = _find_highest_dtype_filtered(args, is_integer_dtype)
        result_dtype = torch.long if result_dtype is None else result_dtype
    else:
        result_dtype = torch.bool
    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
        return (get_computation_dtype(result_dtype), result_dtype)
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH:
        return (result_dtype, result_dtype)
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
        if is_integer_dtype(result_dtype) or is_boolean_dtype(result_dtype):
            result_dtype = torch.get_default_dtype()
        return (get_computation_dtype(result_dtype), result_dtype)
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
        computation_dtype = get_computation_dtype(result_dtype)
        if is_complex_dtype(result_dtype):
            result_dtype = corresponding_real_dtype(result_dtype)
        return (computation_dtype, result_dtype)
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG:
        if is_boolean_dtype(result_dtype):
            return (torch.long, torch.long)
        return (get_computation_dtype(result_dtype), result_dtype)
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return (get_computation_dtype(result_dtype), torch.bool)
    else:
        raise ValueError(f'Unknown type promotion kind {str(type_promotion_kind)}')