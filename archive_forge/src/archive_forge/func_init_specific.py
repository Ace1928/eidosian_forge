import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
def init_specific(self, context, builder, arrty, arr):
    zero = context.get_constant(types.intp, 0)
    data = arr.data
    ndim = arrty.ndim
    shapes = cgutils.unpack_tuple(builder, arr.shape, ndim)
    indices = cgutils.alloca_once(builder, zero.type, size=context.get_constant(types.intp, arrty.ndim))
    pointers = cgutils.alloca_once(builder, data.type, size=context.get_constant(types.intp, arrty.ndim))
    exhausted = cgutils.alloca_once_value(builder, cgutils.false_byte)
    for dim in range(ndim):
        idxptr = cgutils.gep_inbounds(builder, indices, dim)
        ptrptr = cgutils.gep_inbounds(builder, pointers, dim)
        builder.store(data, ptrptr)
        builder.store(zero, idxptr)
        dim_size = shapes[dim]
        dim_is_empty = builder.icmp_unsigned('==', dim_size, zero)
        with cgutils.if_unlikely(builder, dim_is_empty):
            builder.store(cgutils.true_byte, exhausted)
    self.indices = indices
    self.pointers = pointers
    self.exhausted = exhausted