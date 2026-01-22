import operator
from numba import njit, literally
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError
from numba.core.extending import lower_builtin
from numba.core.extending import models, register_model
from numba.core.extending import make_attribute_wrapper
from numba.core.extending import type_callable
from numba.core.extending import overload
from numba.core.extending import typeof_impl
import unittest
@lower_builtin(Dummy, types.intp)
def impl_dummy(context, builder, sig, args):
    typ = sig.return_type
    [value] = args
    dummy = cgutils.create_struct_proxy(typ)(context, builder)
    dummy.value = value
    return dummy._getvalue()