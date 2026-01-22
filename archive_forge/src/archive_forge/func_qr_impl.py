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
def qr_impl(a):
    n = a.shape[-1]
    m = a.shape[-2]
    if n == 0 or m == 0:
        raise np.linalg.LinAlgError('Arrays cannot be empty')
    _check_finite_matrix(a)
    q = _copy_to_fortran_order(a)
    lda = m
    minmn = min(m, n)
    tau = np.empty(minmn, dtype=a.dtype)
    ret = numba_ez_geqrf(kind, m, n, q.ctypes, m, tau.ctypes)
    if ret < 0:
        fatal_error_func()
        assert 0
    r = np.zeros((n, minmn), dtype=a.dtype).T
    for i in range(minmn):
        for j in range(i + 1):
            r[j, i] = q[j, i]
    for i in range(minmn, n):
        for j in range(minmn):
            r[j, i] = q[j, i]
    ret = numba_ez_xxgqr(kind, m, minmn, minmn, q.ctypes, m, tau.ctypes)
    _handle_err_maybe_convergence_problem(ret)
    _dummy_liveness_func([tau.size, q.size])
    return (q[:, :minmn], r)