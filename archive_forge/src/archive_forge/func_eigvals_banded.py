import warnings
import numpy
from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def eigvals_banded(a_band, lower=False, overwrite_a_band=False, select='a', select_range=None, check_finite=True):
    """
    Solve real symmetric or complex Hermitian band matrix eigenvalue problem.

    Find eigenvalues w of a::

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
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
        band matrices
    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
        matrices
    eigvals : eigenvalues of general arrays
    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
    eig : eigenvalues and right eigenvectors for non-symmetric arrays

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigvals_banded
    >>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
    >>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
    >>> w = eigvals_banded(Ab, lower=True)
    >>> w
    array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])
    """
    return eig_banded(a_band, lower=lower, eigvals_only=1, overwrite_a_band=overwrite_a_band, select=select, select_range=select_range, check_finite=check_finite)