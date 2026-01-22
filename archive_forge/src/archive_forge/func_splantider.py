import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
def splantider(tck, n=1):
    """
    Compute the spline for the antiderivative (integral) of a given spline.

    Parameters
    ----------
    tck : tuple of (t, c, k)
        Spline whose antiderivative to compute
    n : int, optional
        Order of antiderivative to evaluate. Default: 1

    Returns
    -------
    tck_ader : tuple of (t2, c2, k2)
        Spline of order k2=k+n representing the antiderivative of the input
        spline.

    See Also
    --------
    splder, splev, spalde

    Notes
    -----
    The `splder` function is the inverse operation of this function.
    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
    rounding error.

    .. seealso:: :class:`scipy.interpolate.splantider`
    """
    if n < 0:
        return splder(tck, -n)
    t, c, k = tck
    sh = (slice(None),) + (None,) * len(c.shape[1:])
    for j in range(n):
        dt = t[k + 1:] - t[:-k - 1]
        dt = dt[sh]
        c = cupy.cumsum(c[:-k - 1] * dt, axis=0) / (k + 1)
        c = cupy.r_[cupy.zeros((1,) + c.shape[1:]), c, [c[-1]] * (k + 2)]
        t = cupy.r_[t[0], t, t[-1]]
        k += 1
    return (t, c, k)