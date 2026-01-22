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
def sv_function(a):
    """
        Computes singular values.
        """
    n = a.shape[-1]
    m = a.shape[-2]
    if m == 0 or n == 0:
        raise np.linalg.LinAlgError('Arrays cannot be empty')
    _check_finite_matrix(a)
    ldu = m
    minmn = min(m, n)
    ucol = 1
    ldvt = 1
    acpy = _copy_to_fortran_order(a)
    s = np.empty(minmn, dtype=np_ret_type)
    r = numba_ez_gesdd(kind, JOBZ_N, m, n, acpy.ctypes, m, s.ctypes, u.ctypes, ldu, vt.ctypes, ldvt)
    _handle_err_maybe_convergence_problem(r)
    _dummy_liveness_func([acpy.size, vt.size, u.size, s.size])
    return s