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
@lower_builtin('static_getitem', types.Record, types.IntegerLiteral)
def record_static_getitem_int(context, builder, sig, args):
    """
    Record.__getitem__ redirects to getattr()
    """
    idx = sig.args[1].literal_value
    fields = list(sig.args[0].fields)
    ll_field = context.insert_const_string(builder.module, fields[idx])
    impl = context.get_getattr(sig.args[0], ll_field)
    return impl(context, builder, sig.args[0], args[0], fields[idx])