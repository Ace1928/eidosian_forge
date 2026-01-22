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
@lower_builtin('static_getitem', types.NumberClass, types.Any)
def static_getitem_number_clazz(context, builder, sig, args):
    """This handles the "static_getitem" when a Numba type is subscripted e.g:
    var = typed.List.empty_list(float64[::1, :])
    It only allows this on simple numerical types. Compound types, like
    records, are not supported.
    """
    retty = sig.return_type
    if isinstance(retty, types.Array):
        res = context.get_value_type(retty)(None)
        return impl_ret_untracked(context, builder, retty, res)
    else:
        msg = 'Unreachable; the definition of __getitem__ on the numba.types.abstract.Type metaclass should prevent access.'
        raise errors.LoweringError(msg)