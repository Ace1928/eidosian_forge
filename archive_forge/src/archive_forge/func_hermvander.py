import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def hermvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = H_i(x),

    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Hermite polynomial.

    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
    array ``V = hermvander(x, n)``, then ``np.dot(V, c)`` and
    ``hermval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Hermite series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo-Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Hermite polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermvander
    >>> x = np.array([-1, 0, 1])
    >>> hermvander(x, 3)
    array([[ 1., -2.,  2.,  4.],
           [ 1.,  0., -2., -0.],
           [ 1.,  2.,  2., -4.]])

    """
    ideg = pu._deprecate_as_int(deg, 'deg')
    if ideg < 0:
        raise ValueError('deg must be non-negative')
    x = np.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        x2 = x * 2
        v[1] = x2
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2] * (2 * (i - 1))
    return np.moveaxis(v, 0, -1)