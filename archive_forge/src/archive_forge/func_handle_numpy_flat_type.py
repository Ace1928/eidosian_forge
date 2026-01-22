from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.NumpyFlatType)
def handle_numpy_flat_type(dmm, ty):
    if ty.array_type.layout == 'C':
        return CContiguousFlatIter(dmm, ty, need_indices=False)
    else:
        return FlatIter(dmm, ty)