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
def solve_circulant(c, b, singular='raise', tol=None, caxis=-1, baxis=0, outaxis=0):
    """Solve C x = b for x, where C is a circulant matrix.

    `C` is the circulant matrix associated with the vector `c`.

    The system is solved by doing division in Fourier space. The
    calculation is::

        x = ifft(fft(b) / fft(c))

    where `fft` and `ifft` are the fast Fourier transform and its inverse,
    respectively. For a large vector `c`, this is *much* faster than
    solving the system with the full circulant matrix.

    Parameters
    ----------
    c : array_like
        The coefficients of the circulant matrix.
    b : array_like
        Right-hand side matrix in ``a x = b``.
    singular : str, optional
        This argument controls how a near singular circulant matrix is
        handled.  If `singular` is "raise" and the circulant matrix is
        near singular, a `LinAlgError` is raised. If `singular` is
        "lstsq", the least squares solution is returned. Default is "raise".
    tol : float, optional
        If any eigenvalue of the circulant matrix has an absolute value
        that is less than or equal to `tol`, the matrix is considered to be
        near singular. If not given, `tol` is set to::

            tol = abs_eigs.max() * abs_eigs.size * np.finfo(np.float64).eps

        where `abs_eigs` is the array of absolute values of the eigenvalues
        of the circulant matrix.
    caxis : int
        When `c` has dimension greater than 1, it is viewed as a collection
        of circulant vectors. In this case, `caxis` is the axis of `c` that
        holds the vectors of circulant coefficients.
    baxis : int
        When `b` has dimension greater than 1, it is viewed as a collection
        of vectors. In this case, `baxis` is the axis of `b` that holds the
        right-hand side vectors.
    outaxis : int
        When `c` or `b` are multidimensional, the value returned by
        `solve_circulant` is multidimensional. In this case, `outaxis` is
        the axis of the result that holds the solution vectors.

    Returns
    -------
    x : ndarray
        Solution to the system ``C x = b``.

    Raises
    ------
    LinAlgError
        If the circulant matrix associated with `c` is near singular.

    See Also
    --------
    circulant : circulant matrix

    Notes
    -----
    For a 1-D vector `c` with length `m`, and an array `b`
    with shape ``(m, ...)``,

        solve_circulant(c, b)

    returns the same result as

        solve(circulant(c), b)

    where `solve` and `circulant` are from `scipy.linalg`.

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import solve_circulant, solve, circulant, lstsq

    >>> c = np.array([2, 2, 4])
    >>> b = np.array([1, 2, 3])
    >>> solve_circulant(c, b)
    array([ 0.75, -0.25,  0.25])

    Compare that result to solving the system with `scipy.linalg.solve`:

    >>> solve(circulant(c), b)
    array([ 0.75, -0.25,  0.25])

    A singular example:

    >>> c = np.array([1, 1, 0, 0])
    >>> b = np.array([1, 2, 3, 4])

    Calling ``solve_circulant(c, b)`` will raise a `LinAlgError`.  For the
    least square solution, use the option ``singular='lstsq'``:

    >>> solve_circulant(c, b, singular='lstsq')
    array([ 0.25,  1.25,  2.25,  1.25])

    Compare to `scipy.linalg.lstsq`:

    >>> x, resid, rnk, s = lstsq(circulant(c), b)
    >>> x
    array([ 0.25,  1.25,  2.25,  1.25])

    A broadcasting example:

    Suppose we have the vectors of two circulant matrices stored in an array
    with shape (2, 5), and three `b` vectors stored in an array with shape
    (3, 5).  For example,

    >>> c = np.array([[1.5, 2, 3, 0, 0], [1, 1, 4, 3, 2]])
    >>> b = np.arange(15).reshape(-1, 5)

    We want to solve all combinations of circulant matrices and `b` vectors,
    with the result stored in an array with shape (2, 3, 5). When we
    disregard the axes of `c` and `b` that hold the vectors of coefficients,
    the shapes of the collections are (2,) and (3,), respectively, which are
    not compatible for broadcasting. To have a broadcast result with shape
    (2, 3), we add a trivial dimension to `c`: ``c[:, np.newaxis, :]`` has
    shape (2, 1, 5). The last dimension holds the coefficients of the
    circulant matrices, so when we call `solve_circulant`, we can use the
    default ``caxis=-1``. The coefficients of the `b` vectors are in the last
    dimension of the array `b`, so we use ``baxis=-1``. If we use the
    default `outaxis`, the result will have shape (5, 2, 3), so we'll use
    ``outaxis=-1`` to put the solution vectors in the last dimension.

    >>> x = solve_circulant(c[:, np.newaxis, :], b, baxis=-1, outaxis=-1)
    >>> x.shape
    (2, 3, 5)
    >>> np.set_printoptions(precision=3)  # For compact output of numbers.
    >>> x
    array([[[-0.118,  0.22 ,  1.277, -0.142,  0.302],
            [ 0.651,  0.989,  2.046,  0.627,  1.072],
            [ 1.42 ,  1.758,  2.816,  1.396,  1.841]],
           [[ 0.401,  0.304,  0.694, -0.867,  0.377],
            [ 0.856,  0.758,  1.149, -0.412,  0.831],
            [ 1.31 ,  1.213,  1.603,  0.042,  1.286]]])

    Check by solving one pair of `c` and `b` vectors (cf. ``x[1, 1, :]``):

    >>> solve_circulant(c[1], b[1, :])
    array([ 0.856,  0.758,  1.149, -0.412,  0.831])

    """
    c = np.atleast_1d(c)
    nc = _get_axis_len('c', c, caxis)
    b = np.atleast_1d(b)
    nb = _get_axis_len('b', b, baxis)
    if nc != nb:
        raise ValueError(f'Shapes of c {c.shape} and b {b.shape} are incompatible')
    fc = np.fft.fft(np.moveaxis(c, caxis, -1), axis=-1)
    abs_fc = np.abs(fc)
    if tol is None:
        tol = abs_fc.max(axis=-1) * nc * np.finfo(np.float64).eps
        if tol.shape != ():
            tol.shape = tol.shape + (1,)
        else:
            tol = np.atleast_1d(tol)
    near_zeros = abs_fc <= tol
    is_near_singular = np.any(near_zeros)
    if is_near_singular:
        if singular == 'raise':
            raise LinAlgError('near singular circulant matrix.')
        else:
            fc[near_zeros] = 1
    fb = np.fft.fft(np.moveaxis(b, baxis, -1), axis=-1)
    q = fb / fc
    if is_near_singular:
        mask = np.ones_like(b, dtype=bool) & near_zeros
        q[mask] = 0
    x = np.fft.ifft(q, axis=-1)
    if not (np.iscomplexobj(c) or np.iscomplexobj(b)):
        x = x.real
    if outaxis != -1:
        x = np.moveaxis(x, -1, outaxis)
    return x