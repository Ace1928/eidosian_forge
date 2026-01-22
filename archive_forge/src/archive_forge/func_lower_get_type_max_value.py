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
@lower_builtin(get_type_max_value, types.NumberClass)
@lower_builtin(get_type_max_value, types.DType)
def lower_get_type_max_value(context, builder, sig, args):
    typ = sig.args[0].dtype
    if isinstance(typ, types.Integer):
        bw = typ.bitwidth
        lty = ir.IntType(bw)
        val = typ.maxval
        res = ir.Constant(lty, val)
    elif isinstance(typ, types.Float):
        bw = typ.bitwidth
        if bw == 32:
            lty = ir.FloatType()
        elif bw == 64:
            lty = ir.DoubleType()
        else:
            raise NotImplementedError('llvmlite only supports 32 and 64 bit floats')
        npty = getattr(np, 'float{}'.format(bw))
        res = ir.Constant(lty, np.inf)
    elif isinstance(typ, (types.NPDatetime, types.NPTimedelta)):
        bw = 64
        lty = ir.IntType(bw)
        val = types.int64.maxval
        res = ir.Constant(lty, val)
    return impl_ret_untracked(context, builder, lty, res)