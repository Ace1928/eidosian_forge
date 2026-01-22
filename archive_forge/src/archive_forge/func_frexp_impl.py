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
@lower(math.frexp, types.Float)
def frexp_impl(context, builder, sig, args):
    val, = args
    fltty = context.get_data_type(sig.args[0])
    intty = context.get_data_type(sig.return_type[1])
    expptr = cgutils.alloca_once(builder, intty, name='exp')
    fnty = llvmlite.ir.FunctionType(fltty, (fltty, llvmlite.ir.PointerType(intty)))
    fname = {'float': 'numba_frexpf', 'double': 'numba_frexp'}[str(fltty)]
    fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
    res = builder.call(fn, (val, expptr))
    res = cgutils.make_anonymous_struct(builder, (res, builder.load(expptr)))
    return impl_ret_untracked(context, builder, sig.return_type, res)