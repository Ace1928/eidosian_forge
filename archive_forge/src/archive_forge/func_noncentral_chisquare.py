from cupy.random import _generator
from cupy import _util
def noncentral_chisquare(df, nonc, size=None, dtype=float):
    """Noncentral chisquare distribution.

    Returns an array of samples drawn from the noncentral chisquare
    distribution. Its probability density function is defined as

    .. math::
       f(x) = \\frac{1}{2}e^{-(x+\\lambda)/2} \\
        \\left(\\frac{x}{\\lambda}\\right)^{k/4 - 1/2} \\
        I_{k/2 - 1}(\\sqrt{\\lambda x}),

    where :math:`I` is the modified Bessel function of the first kind.

    Args:
        df (float): Parameter of the noncentral chisquare distribution
            :math:`k`.
        nonc (float): Parameter of the noncentral chisquare distribution
            :math:`\\lambda`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the noncentral chisquare distribution.

    .. seealso::
        :func:`numpy.random.noncentral_chisquare`
    """
    rs = _generator.get_random_state()
    return rs.noncentral_chisquare(df, nonc, size=size, dtype=dtype)