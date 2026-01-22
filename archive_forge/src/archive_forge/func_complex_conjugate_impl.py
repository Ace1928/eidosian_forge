import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
@lower_builtin('complex.conjugate', types.Complex)
def complex_conjugate_impl(context, builder, sig, args):
    from numba.cpython import mathimpl
    z = context.make_complex(builder, sig.args[0], args[0])
    z.imag = mathimpl.negate_real(builder, z.imag)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)