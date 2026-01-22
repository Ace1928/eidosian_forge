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
def reduction_dtypes(arg, output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND, dtype: Optional[torch.dtype]=None) -> Tuple[torch.dtype, Optional[torch.dtype]]:
    inp_dtype = dtype if dtype is not None else arg.dtype
    computation_dtype = get_computation_dtype(inp_dtype)
    if output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME or output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT:
        result_dtype = dtype if dtype else arg.dtype
        if output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT and is_complex_dtype(result_dtype):
            result_dtype = corresponding_real_dtype(result_dtype)
    elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE:
        result_dtype = None
    else:
        result_dtype = torch.bool
    return (computation_dtype, result_dtype)