import math
import operator
import sys
import numpy as np
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core.imputils import Registry, impl_ret_untracked
from numba import typeof
from numba.core import types, utils, config, cgutils
from numba.core.extending import overload
from numba.core.typing import signature
from numba.cpython.unsafe.numbers import trailing_zeros
@lower(math.ldexp, types.Float, types.intc)
def ldexp_impl(context, builder, sig, args):
    val, exp = args
    fltty, intty = map(context.get_data_type, sig.args)
    fnty = llvmlite.ir.FunctionType(fltty, (fltty, intty))
    fname = {'float': 'numba_ldexpf', 'double': 'numba_ldexp'}[str(fltty)]
    fn = cgutils.insert_pure_function(builder.module, fnty, name=fname)
    res = builder.call(fn, (val, exp))
    return impl_ret_untracked(context, builder, sig.return_type, res)