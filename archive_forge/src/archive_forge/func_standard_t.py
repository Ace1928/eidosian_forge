from cupy.random import _generator
from cupy import _util
def standard_t(df, size=None, dtype=float):
    """Standard Student's t distribution.

    Returns an array of samples drawn from the standard Student's t
    distribution. Its probability density function is defined as

    .. math::
        f(x) = \\frac{\\Gamma(\\frac{\\nu+1}{2})}             {\\sqrt{\\nu\\pi}\\Gamma(\\frac{\\nu}{2})}             \\left(1 + \\frac{x^2}{\\nu} \\right)^{-(\\frac{\\nu+1}{2})}.

    Args:
        df (float or array_like of floats): Degree of freedom :math:`\\nu`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard Student's t distribution.

    .. seealso::
        :func:`numpy.random.standard_t`
    """
    rs = _generator.get_random_state()
    return rs.standard_t(df, size, dtype)