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
@overload(np.linalg.matrix_rank)
def matrix_rank_impl(A, tol=None):
    """
    Computes rank for matrices and vectors.
    The only issue that may arise is that because numpy uses double
    precision lapack calls whereas numba uses type specific lapack
    calls, some singular values may differ and therefore counting the
    number of them above a tolerance may lead to different counts,
    and therefore rank, in some cases.
    """
    ensure_lapack()
    _check_linalg_1_or_2d_matrix(A, 'matrix_rank')

    def _2d_matrix_rank_impl(A, tol):
        if tol in (None, types.none):
            nb_type = getattr(A.dtype, 'underlying_float', A.dtype)
            np_type = np_support.as_dtype(nb_type)
            eps_val = np.finfo(np_type).eps

            def _2d_tol_none_impl(A, tol=None):
                s = _compute_singular_values(A)
                r = A.shape[0]
                c = A.shape[1]
                l = max(r, c)
                t = s[0] * l * eps_val
                return _get_rank_from_singular_values(s, t)
            return _2d_tol_none_impl
        else:

            def _2d_tol_not_none_impl(A, tol=None):
                s = _compute_singular_values(A)
                return _get_rank_from_singular_values(s, tol)
            return _2d_tol_not_none_impl

    def _get_matrix_rank_impl(A, tol):
        ndim = A.ndim
        if ndim == 1:

            def _1d_matrix_rank_impl(A, tol=None):
                for k in range(len(A)):
                    if A[k] != 0.0:
                        return 1
                return 0
            return _1d_matrix_rank_impl
        elif ndim == 2:
            return _2d_matrix_rank_impl(A, tol)
        else:
            assert 0
    return _get_matrix_rank_impl(A, tol)