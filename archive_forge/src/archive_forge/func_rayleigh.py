from cupy.random import _generator
from cupy import _util
def rayleigh(scale=1.0, size=None, dtype=float):
    """Rayleigh distribution.

    Returns an array of samples drawn from the rayleigh distribution.
    Its probability density function is defined as

      .. math::
         f(x) = \\frac{x}{\\sigma^2}e^{\\frac{-x^2}{2-\\sigma^2}}, x \\ge 0.

    Args:
        scale (array): Parameter of the rayleigh distribution :math:`\\sigma`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the rayleigh distribution.

    .. seealso:: :func:`numpy.random.rayleigh`
    """
    rs = _generator.get_random_state()
    return rs.rayleigh(scale, size, dtype)