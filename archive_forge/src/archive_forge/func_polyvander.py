import cupy
def polyvander(x, deg):
    """Computes the Vandermonde matrix of given degree.

    Args:
        x (cupy.ndarray): array of points
        deg (int): degree of the resulting matrix.

    Returns:
        cupy.ndarray: The Vandermonde matrix

    .. seealso:: :func:`numpy.polynomial.polynomial.polyvander`

    """
    deg = cupy.polynomial.polyutils._deprecate_as_int(deg, 'deg')
    if deg < 0:
        raise ValueError('degree must be non-negative')
    if x.ndim == 0:
        x = x.ravel()
    dtype = cupy.float64 if x.dtype.kind in 'biu' else x.dtype
    out = x ** cupy.arange(deg + 1, dtype=dtype).reshape((-1,) + (1,) * x.ndim)
    return cupy.moveaxis(out, 0, -1)