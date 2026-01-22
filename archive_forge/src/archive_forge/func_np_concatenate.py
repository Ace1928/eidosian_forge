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
def np_concatenate(typingctx, arrays, axis):
    ret = np_concatenate_typer(typingctx, arrays, axis)
    assert isinstance(ret, types.Array)
    sig = ret(arrays, axis)

    def codegen(context, builder, sig, args):
        axis = context.cast(builder, args[1], sig.args[1], types.intp)
        return _np_concatenate(context, builder, list(sig.args[0]), cgutils.unpack_tuple(builder, args[0]), sig.return_type, axis)
    return (sig, codegen)