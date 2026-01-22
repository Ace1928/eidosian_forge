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
@lower_builtin(round, types.Float)
def round_impl_unary(context, builder, sig, args):
    fltty = sig.args[0]
    llty = context.get_value_type(fltty)
    module = builder.module
    fnty = ir.FunctionType(llty, [llty])
    fn = cgutils.get_or_insert_function(module, fnty, _round_intrinsic(fltty))
    res = builder.call(fn, args)
    res = builder.fptosi(res, context.get_value_type(sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)