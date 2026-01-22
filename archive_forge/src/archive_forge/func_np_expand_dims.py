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
@intrinsic
def np_expand_dims(typingctx, a, axis):
    layout = a.layout if a.ndim <= 1 else 'A'
    ret = a.copy(ndim=a.ndim + 1, layout=layout)
    sig = ret(a, axis)

    def codegen(context, builder, sig, args):
        axis = context.cast(builder, args[1], sig.args[1], types.intp)
        axis = _normalize_axis(context, builder, 'np.expand_dims', sig.return_type.ndim, axis)
        ret = expand_dims(context, builder, sig, args, axis)
        return impl_ret_borrowed(context, builder, sig.return_type, ret)
    return (sig, codegen)