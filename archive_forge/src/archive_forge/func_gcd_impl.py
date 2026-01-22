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
def gcd_impl(context, builder, sig, args):
    xty, yty = sig.args
    assert xty == yty == sig.return_type
    x, y = args

    def gcd(a, b):
        """
        Stein's algorithm, heavily cribbed from Julia implementation.
        """
        T = type(a)
        if a == 0:
            return abs(b)
        if b == 0:
            return abs(a)
        za = trailing_zeros(a)
        zb = trailing_zeros(b)
        k = min(za, zb)
        u = _unsigned(abs(np.right_shift(a, za)))
        v = _unsigned(abs(np.right_shift(b, zb)))
        while u != v:
            if u > v:
                u, v = (v, u)
            v -= u
            v = np.right_shift(v, trailing_zeros(v))
        r = np.left_shift(T(u), k)
        return r
    res = context.compile_internal(builder, gcd, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)