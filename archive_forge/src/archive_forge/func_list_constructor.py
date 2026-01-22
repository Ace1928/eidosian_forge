import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
@lower_builtin(list)
def list_constructor(context, builder, sig, args):
    list_type = sig.return_type
    list_len = 0
    inst = ListInstance.allocate(context, builder, list_type, list_len)
    return impl_ret_new_ref(context, builder, list_type, inst.value)