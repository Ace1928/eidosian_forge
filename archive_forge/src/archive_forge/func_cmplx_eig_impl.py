import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def cmplx_eig_impl(a):
    """
        eig() implementation for complex arrays.
        """
    n = a.shape[-1]
    if a.shape[-2] != n:
        msg = 'Last 2 dimensions of the array must be square.'
        raise np.linalg.LinAlgError(msg)
    _check_finite_matrix(a)
    acpy = _copy_to_fortran_order(a)
    ldvl = 1
    ldvr = n
    w = np.empty(n, dtype=a.dtype)
    vl = np.empty((n, ldvl), dtype=a.dtype)
    vr = np.empty((n, ldvr), dtype=a.dtype)
    if n == 0:
        return (w, vr.T)
    r = numba_ez_cgeev(kind, JOBVL, JOBVR, n, acpy.ctypes, n, w.ctypes, vl.ctypes, ldvl, vr.ctypes, ldvr)
    _handle_err_maybe_convergence_problem(r)
    _dummy_liveness_func([acpy.size, vl.size, vr.size, w.size])
    return (w, vr.T)