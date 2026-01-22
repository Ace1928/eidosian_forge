import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
def set_pop(context, builder, sig, args):
    inst = SetInstance(context, builder, sig.args[0], args[0])
    used = inst.payload.used
    with builder.if_then(cgutils.is_null(builder, used), likely=False):
        context.call_conv.return_user_exc(builder, KeyError, ('set.pop(): empty set',))
    return inst.pop()