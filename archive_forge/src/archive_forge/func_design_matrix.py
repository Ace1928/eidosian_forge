import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
@classmethod
def design_matrix(cls, x, t, k, extrapolate=False):
    """
        Returns a design matrix as a CSR format sparse array.

        Parameters
        ----------
        x : array_like, shape (n,)
            Points to evaluate the spline at.
        t : array_like, shape (nt,)
            Sorted 1D array of knots.
        k : int
            B-spline degree.
        extrapolate : bool or 'periodic', optional
            Whether to extrapolate based on the first and last intervals
            or raise an error. If 'periodic', periodic extrapolation is used.
            Default is False.

        Returns
        -------
        design_matrix : `csr_matrix` object
            Sparse matrix in CSR format where each row contains all the basis
            elements of the input row (first row = basis elements of x[0],
            ..., last row = basis elements x[-1]).

        Notes
        -----
        In each row of the design matrix all the basis elements are evaluated
        at the certain point (first row - x[0], ..., last row - x[-1]).
        `nt` is a length of the vector of knots: as far as there are
        `nt - k - 1` basis elements, `nt` should be not less than `2 * k + 2`
        to have at least `k + 1` basis element.

        Out of bounds `x` raises a ValueError.

        .. note::
            This method returns a `csr_matrix` instance as CuPy still does not
            have `csr_array`.

        .. seealso:: :class:`scipy.interpolate.BSpline`
        """
    x = _as_float_array(x, True)
    t = _as_float_array(t, True)
    if extrapolate != 'periodic':
        extrapolate = bool(extrapolate)
    if k < 0:
        raise ValueError('Spline order cannot be negative.')
    if t.ndim != 1 or np.any(t[1:] < t[:-1]):
        raise ValueError(f'Expect t to be a 1-D sorted array_like, but got t={t}.')
    if len(t) < 2 * k + 2:
        raise ValueError(f'Length t is not enough for k={k}.')
    if extrapolate == 'periodic':
        n = t.size - k - 1
        x = t[k] + (x - t[k]) % (t[n] - t[k])
        extrapolate = False
    elif not extrapolate and (min(x) < t[k] or max(x) > t[t.shape[0] - k - 1]):
        raise ValueError(f'Out of bounds w/ x = {x}.')
    n = x.shape[0]
    nnz = n * (k + 1)
    if nnz < cupy.iinfo(cupy.int32).max:
        int_dtype = cupy.int32
    else:
        int_dtype = cupy.int64
    indices = cupy.empty(n * (k + 1), dtype=int_dtype)
    indptr = cupy.arange(0, (n + 1) * (k + 1), k + 1, dtype=int_dtype)
    data, indices = _make_design_matrix(x, t, k, extrapolate, indices)
    return csr_matrix((data, indices, indptr), shape=(x.shape[0], t.shape[0] - k - 1))