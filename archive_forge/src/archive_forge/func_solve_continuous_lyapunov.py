import warnings
import numpy as np
from numpy.linalg import inv, LinAlgError, norm, cond, svd
from ._basic import solve, solve_triangular, matrix_balance
from .lapack import get_lapack_funcs
from ._decomp_schur import schur
from ._decomp_lu import lu
from ._decomp_qr import qr
from ._decomp_qz import ordqz
from ._decomp import _asarray_validated
from ._special_matrices import kron, block_diag
def solve_continuous_lyapunov(a, q):
    """
    Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.

    Uses the Bartels-Stewart algorithm to find :math:`X`.

    Parameters
    ----------
    a : array_like
        A square matrix

    q : array_like
        Right-hand side square matrix

    Returns
    -------
    x : ndarray
        Solution to the continuous Lyapunov equation

    See Also
    --------
    solve_discrete_lyapunov : computes the solution to the discrete-time
        Lyapunov equation
    solve_sylvester : computes the solution to the Sylvester equation

    Notes
    -----
    The continuous Lyapunov equation is a special form of the Sylvester
    equation, hence this solver relies on LAPACK routine ?TRSYL.

    .. versionadded:: 0.11.0

    Examples
    --------
    Given `a` and `q` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])
    >>> b = np.array([2, 4, -1])
    >>> q = np.eye(3)
    >>> x = linalg.solve_continuous_lyapunov(a, q)
    >>> x
    array([[ -0.75  ,   0.875 ,  -3.75  ],
           [  0.875 ,  -1.375 ,   5.3125],
           [ -3.75  ,   5.3125, -27.0625]])
    >>> np.allclose(a.dot(x) + x.dot(a.T), q)
    True
    """
    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))
    r_or_c = float
    for ind, _ in enumerate((a, q)):
        if np.iscomplexobj(_):
            r_or_c = complex
        if not np.equal(*_.shape):
            raise ValueError('Matrix {} should be square.'.format('aq'[ind]))
    if a.shape != q.shape:
        raise ValueError('Matrix a and q should have the same shape.')
    r, u = schur(a, output='real')
    f = u.conj().T.dot(q.dot(u))
    trsyl = get_lapack_funcs('trsyl', (r, f))
    dtype_string = 'T' if r_or_c == float else 'C'
    y, scale, info = trsyl(r, r, f, tranb=dtype_string)
    if info < 0:
        raise ValueError(f'?TRSYL exited with the internal error "illegal value in argument number {-info}.". See LAPACK documentation for the ?TRSYL error codes.')
    elif info == 1:
        warnings.warn('Input "a" has an eigenvalue pair whose sum is very close to or exactly zero. The solution is obtained via perturbing the coefficients.', RuntimeWarning, stacklevel=2)
    y *= scale
    return u.dot(y).dot(u.conj().T)