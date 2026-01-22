import functools
import warnings
import numpy
import cupy
from cupy import _core
def corrcoef(a, y=None, rowvar=True, bias=None, ddof=None, *, dtype=None):
    """Returns the Pearson product-moment correlation coefficients of an array.

    Args:
        a (cupy.ndarray): Array to compute the Pearson product-moment
            correlation coefficients.
        y (cupy.ndarray): An additional set of variables and observations.
        rowvar (bool): If ``True``, then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is
            transposed.
        bias (None): Has no effect, do not use.
        ddof (None): Has no effect, do not use.
        dtype: Data type specifier. By default, the return data-type will have
            at least `numpy.float64` precision.

    Returns:
        cupy.ndarray: The Pearson product-moment correlation coefficients of
        the input array.

    .. seealso:: :func:`numpy.corrcoef`

    """
    if bias is not None or ddof is not None:
        warnings.warn('bias and ddof have no effect and are deprecated', DeprecationWarning)
    out = cov(a, y, rowvar, dtype=dtype)
    try:
        d = cupy.diag(out)
    except ValueError:
        return out / out
    stddev = cupy.sqrt(d.real)
    out /= stddev[:, None]
    out /= stddev[None, :]
    cupy.clip(out.real, -1, 1, out=out.real)
    if cupy.iscomplexobj(out):
        cupy.clip(out.imag, -1, 1, out=out.imag)
    return out