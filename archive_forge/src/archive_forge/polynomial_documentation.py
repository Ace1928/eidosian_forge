import cupy
Computes the companion matrix of c.

    Args:
        c (cupy.ndarray): 1-D array of polynomial coefficients
            ordered from low to high degree.

    Returns:
        cupy.ndarray: Companion matrix of dimensions (deg, deg).

    .. seealso:: :func:`numpy.polynomial.polynomial.polycompanion`

    