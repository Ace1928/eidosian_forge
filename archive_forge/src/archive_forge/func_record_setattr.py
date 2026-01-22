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
@lower_setattr_generic(types.Record)
def record_setattr(context, builder, sig, args, attr):
    """
    Generic setattr() implementation for records: set the given record member.
    """
    typ, valty = sig.args
    target, val = args
    context.sentry_record_alignment(typ, attr)
    offset = typ.offset(attr)
    elemty = typ.typeof(attr)
    if isinstance(elemty, types.NestedArray):
        val_struct = cgutils.create_struct_proxy(valty)(context, builder, value=args[1])
        src = val_struct.data
        dest = cgutils.get_record_member(builder, target, offset, src.type.pointee)
        cgutils.memcpy(builder, dest, src, context.get_constant(types.intp, elemty.nitems))
    else:
        dptr = cgutils.get_record_member(builder, target, offset, context.get_data_type(elemty))
        val = context.cast(builder, val, valty, elemty)
        align = None if typ.aligned else 1
        context.pack_value(builder, elemty, val, dptr, align=align)