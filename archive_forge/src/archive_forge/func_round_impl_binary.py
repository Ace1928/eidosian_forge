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
@lower_builtin(round, types.Float, types.Integer)
def round_impl_binary(context, builder, sig, args):
    fltty = sig.args[0]
    _round = types.ExternalFunction(_round_intrinsic(fltty), typing.signature(fltty, fltty))

    def round_ndigits(x, ndigits):
        if math.isinf(x) or math.isnan(x):
            return x
        if ndigits >= 0:
            if ndigits > 22:
                pow1 = 10.0 ** (ndigits - 22)
                pow2 = 1e+22
            else:
                pow1 = 10.0 ** ndigits
                pow2 = 1.0
            y = x * pow1 * pow2
            if math.isinf(y):
                return x
            return _round(y) / pow2 / pow1
        else:
            pow1 = 10.0 ** (-ndigits)
            y = x / pow1
            return _round(y) * pow1
    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)