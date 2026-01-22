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
def np_cfarray(context, builder, sig, args):
    """
    numba.numpy_support.carray(...) and
    numba.numpy_support.farray(...).
    """
    ptrty, shapety = sig.args[:2]
    ptr, shape = args[:2]
    aryty = sig.return_type
    assert aryty.layout in 'CF'
    out_ary = make_array(aryty)(context, builder)
    itemsize = get_itemsize(context, aryty)
    ll_itemsize = cgutils.intp_t(itemsize)
    if isinstance(shapety, types.BaseTuple):
        shapes = cgutils.unpack_tuple(builder, shape)
    else:
        shapety = (shapety,)
        shapes = (shape,)
    shapes = [context.cast(builder, value, fromty, types.intp) for fromty, value in zip(shapety, shapes)]
    off = ll_itemsize
    strides = []
    if aryty.layout == 'F':
        for s in shapes:
            strides.append(off)
            off = builder.mul(off, s)
    else:
        for s in reversed(shapes):
            strides.append(off)
            off = builder.mul(off, s)
        strides.reverse()
    data = builder.bitcast(ptr, context.get_data_type(aryty.dtype).as_pointer())
    populate_array(out_ary, data=data, shape=shapes, strides=strides, itemsize=ll_itemsize, meminfo=None)
    res = out_ary._getvalue()
    return impl_ret_new_ref(context, builder, sig.return_type, res)