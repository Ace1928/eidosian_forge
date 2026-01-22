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
@lower(math.pow, types.Float, types.Float)
@lower(math.pow, types.Float, types.Integer)
def pow_impl(context, builder, sig, args):
    impl = context.get_function(operator.pow, sig)
    return impl(builder, args)