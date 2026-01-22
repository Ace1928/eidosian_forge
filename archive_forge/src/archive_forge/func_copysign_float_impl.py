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
@lower(math.copysign, types.Float, types.Float)
def copysign_float_impl(context, builder, sig, args):
    lty = args[0].type
    mod = builder.module
    fn = cgutils.get_or_insert_function(mod, llvmlite.ir.FunctionType(lty, (lty, lty)), 'llvm.copysign.%s' % lty.intrinsic_name)
    res = builder.call(fn, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)