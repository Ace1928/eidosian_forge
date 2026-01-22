import warnings
import numpy
from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def eig_banded(a_band, lower=False, eigvals_only=False, overwrite_a_band=False, select='a', select_range=None, max_ev=0, check_finite=True):
    """
    Solve real symmetric or complex Hermitian band matrix eigenvalue problem.

    Find eigenvalues w and optionally right eigenvectors v of a::

        a v[:,i] = w[i] v[:,i]
        v.H v    = identity

    The matrix a is stored in a_band either in lower diagonal or upper
    diagonal ordered form:

        a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)

    where u is the number of bands above the diagonal.

    Example of a_band (shape of a is (6,6), u=2)::

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
    a_band : (u+1, M) array_like
        The bands of the M by M matrix a.
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    eigvals_only : bool, optional
        Compute only the eigenvalues and no eigenvectors.
        (Default: calculate also eigenvectors)
    overwrite_a_band : bool, optional
        Discard data in a_band (may enhance performance)
    select : {'a', 'v', 'i'}, optional
        Which eigenvalues to calculate

        ======  ========================================
        select  calculated
        ======  ========================================
        'a'     All eigenvalues
        'v'     Eigenvalues in the interval (min, max]
        'i'     Eigenvalues with indices min <= i <= max
        ======  ========================================
    select_range : (min, max), optional
        Range of selected eigenvalues
    max_ev : int, optional
        For select=='v', maximum number of eigenvalues expected.
        For other values of select, has no meaning.

        In doubt, leave this parameter untouched.

    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.
    v : (M, M) float or complex ndarray
        The normalized eigenvector corresponding to the eigenvalue w[i] is
        the column v[:,i]. Only returned if ``eigvals_only=False``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
    eig : eigenvalues and right eigenvectors of general arrays.
    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
    eigh_tridiagonal : eigenvalues and right eigenvectors for
        symmetric/Hermitian tridiagonal matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eig_banded
    >>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
    >>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
    >>> w, v = eig_banded(Ab, lower=True)
    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
    True
    >>> w = eig_banded(Ab, lower=True, eigvals_only=True)
    >>> w
    array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])

    Request only the eigenvalues between ``[-3, 4]``

    >>> w, v = eig_banded(Ab, lower=True, select='v', select_range=[-3, 4])
    >>> w
    array([-2.22987175,  3.95222349])

    """
    if eigvals_only or overwrite_a_band:
        a1 = _asarray_validated(a_band, check_finite=check_finite)
        overwrite_a_band = overwrite_a_band or _datacopied(a1, a_band)
    else:
        a1 = array(a_band)
        if issubclass(a1.dtype.type, inexact) and (not isfinite(a1).all()):
            raise ValueError('array must not contain infs or NaNs')
        overwrite_a_band = 1
    if len(a1.shape) != 2:
        raise ValueError('expected a 2-D array')
    select, vl, vu, il, iu, max_ev = _check_select(select, select_range, max_ev, a1.shape[1])
    del select_range
    if select == 0:
        if a1.dtype.char in 'GFD':
            internal_name = 'hbevd'
        else:
            internal_name = 'sbevd'
        bevd, = get_lapack_funcs((internal_name,), (a1,))
        w, v, info = bevd(a1, compute_v=not eigvals_only, lower=lower, overwrite_ab=overwrite_a_band)
    else:
        if eigvals_only:
            max_ev = 1
        if a1.dtype.char in 'fF':
            lamch, = get_lapack_funcs(('lamch',), (array(0, dtype='f'),))
        else:
            lamch, = get_lapack_funcs(('lamch',), (array(0, dtype='d'),))
        abstol = 2 * lamch('s')
        if a1.dtype.char in 'GFD':
            internal_name = 'hbevx'
        else:
            internal_name = 'sbevx'
        bevx, = get_lapack_funcs((internal_name,), (a1,))
        w, v, m, ifail, info = bevx(a1, vl, vu, il, iu, compute_v=not eigvals_only, mmax=max_ev, range=select, lower=lower, overwrite_ab=overwrite_a_band, abstol=abstol)
        w = w[:m]
        if not eigvals_only:
            v = v[:, :m]
    _check_info(info, internal_name)
    if eigvals_only:
        return w
    return (w, v)