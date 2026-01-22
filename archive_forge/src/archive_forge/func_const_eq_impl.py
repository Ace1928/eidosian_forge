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
@lower_builtin(operator.eq, types.Literal, types.Literal)
@lower_builtin(operator.eq, types.IntegerLiteral, types.IntegerLiteral)
def const_eq_impl(context, builder, sig, args):
    arg1, arg2 = sig.args
    val = 0
    if arg1.literal_value == arg2.literal_value:
        val = 1
    res = ir.Constant(ir.IntType(1), val)
    return impl_ret_untracked(context, builder, sig.return_type, res)