from collections import namedtuple
import math
from functools import reduce
import numpy as np
import operator
import warnings
from llvmlite import ir
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, cgutils
from numba.core.extending import overload, intrinsic
from numba.core.typeconv import Conversion
from numba.core.errors import (TypingError, LoweringError,
from numba.misc.special import literal_unroll
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.typing.builtins import IndexValue, IndexValueType
from numba.extending import overload, register_jitable
@lower_builtin(operator.is_, types.Boolean, types.Boolean)
def bool_is_impl(context, builder, sig, args):
    """
    Implementation for `x is y` for types derived from types.Boolean
    (e.g. BooleanLiteral), and cross-checks between literal and non-literal
    booleans, to satisfy Python's behavior preserving identity for bools.
    """
    arg1, arg2 = args
    arg1_type, arg2_type = sig.args
    _arg1 = context.cast(builder, arg1, arg1_type, types.boolean)
    _arg2 = context.cast(builder, arg2, arg2_type, types.boolean)
    eq_impl = context.get_function(operator.eq, typing.signature(types.boolean, types.boolean, types.boolean))
    return eq_impl(builder, (_arg1, _arg2))