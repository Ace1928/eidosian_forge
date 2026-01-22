from warnings import warn
from itertools import product
import numpy as np
from numpy import atleast_1d, atleast_2d
from .lapack import get_lapack_funcs, _compute_lwork
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from ._decomp import _asarray_validated
from . import _decomp, _decomp_svd
from ._solve_toeplitz import levinson
from ._cythonized_array_utils import find_det_from_lu
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy.linalg._flinalg_py import get_flinalg_funcs  # noqa: F401
def solveh_banded(ab, b, overwrite_ab=False, overwrite_b=False, lower=False, check_finite=True):
    """
    Solve equation a x = b. a is Hermitian positive-definite banded matrix.

    Uses Thomas' Algorithm, which is more efficient than standard LU
    factorization, but should only be used for Hermitian positive-definite
    matrices.

    The matrix ``a`` is stored in `ab` either in lower diagonal or upper
    diagonal ordered form:

        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

    Example of `ab` (shape of ``a`` is (6, 6), number of upper diagonals,
    ``u`` =2)::

        upper form:
        *   *   a02 a13 a24 a35
        *   a01 a12 a23 a34 a45
        a00 a11 a22 a33 a44 a55

        lower form:
        a00 a11 a22 a33 a44 a55
        a10 a21 a32 a43 a54 *
        a20 a31 a42 a53 *   *

    Cells marked with * are not used.

    Parameters
    ----------
    ab : (``u`` + 1, M) array_like
        Banded matrix
    b : (M,) or (M, K) array_like
        Right-hand side
    overwrite_ab : bool, optional
        Discard data in `ab` (may enhance performance)
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance)
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system ``a x = b``. Shape of return matches shape
        of `b`.

    Notes
    -----
    In the case of a non-positive definite matrix ``a``, the solver
    `solve_banded` may be used.

    Examples
    --------
    Solve the banded system ``A x = b``, where::

            [ 4  2 -1  0  0  0]       [1]
            [ 2  5  2 -1  0  0]       [2]
        A = [-1  2  6  2 -1  0]   b = [2]
            [ 0 -1  2  7  2 -1]       [3]
            [ 0  0 -1  2  8  2]       [3]
            [ 0  0  0 -1  2  9]       [3]

    >>> import numpy as np
    >>> from scipy.linalg import solveh_banded

    ``ab`` contains the main diagonal and the nonzero diagonals below the
    main diagonal. That is, we use the lower form:

    >>> ab = np.array([[ 4,  5,  6,  7, 8, 9],
    ...                [ 2,  2,  2,  2, 2, 0],
    ...                [-1, -1, -1, -1, 0, 0]])
    >>> b = np.array([1, 2, 2, 3, 3, 3])
    >>> x = solveh_banded(ab, b, lower=True)
    >>> x
    array([ 0.03431373,  0.45938375,  0.05602241,  0.47759104,  0.17577031,
            0.34733894])


    Solve the Hermitian banded system ``H x = b``, where::

            [ 8   2-1j   0     0  ]        [ 1  ]
        H = [2+1j  5     1j    0  ]    b = [1+1j]
            [ 0   -1j    9   -2-1j]        [1-2j]
            [ 0    0   -2+1j   6  ]        [ 0  ]

    In this example, we put the upper diagonals in the array ``hb``:

    >>> hb = np.array([[0, 2-1j, 1j, -2-1j],
    ...                [8,  5,    9,   6  ]])
    >>> b = np.array([1, 1+1j, 1-2j, 0])
    >>> x = solveh_banded(hb, b)
    >>> x
    array([ 0.07318536-0.02939412j,  0.11877624+0.17696461j,
            0.10077984-0.23035393j, -0.00479904-0.09358128j])

    """
    a1 = _asarray_validated(ab, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError('shapes of ab and b are not compatible.')
    overwrite_b = overwrite_b or _datacopied(b1, b)
    overwrite_ab = overwrite_ab or _datacopied(a1, ab)
    if a1.shape[0] == 2:
        ptsv, = get_lapack_funcs(('ptsv',), (a1, b1))
        if lower:
            d = a1[0, :].real
            e = a1[1, :-1]
        else:
            d = a1[1, :].real
            e = a1[0, 1:].conj()
        d, du, x, info = ptsv(d, e, b1, overwrite_ab, overwrite_ab, overwrite_b)
    else:
        pbsv, = get_lapack_funcs(('pbsv',), (a1, b1))
        c, x, info = pbsv(a1, b1, lower=lower, overwrite_ab=overwrite_ab, overwrite_b=overwrite_b)
    if info > 0:
        raise LinAlgError('%dth leading minor not positive definite' % info)
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal pbsv' % -info)
    return x