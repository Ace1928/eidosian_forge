from cupy.random import _generator
from cupy import _util
def weibull(a, size=None, dtype=float):
    """weibull distribution.

    Returns an array of samples drawn from the weibull distribution. Its
    probability density function is defined as

    .. math::
       f(x) = ax^{(a-1)}e^{-x^a}.

    Args:
        a (float): Parameter of the weibull distribution :math:`a`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the weibull distribution.

    .. seealso::
        :func:`numpy.random.weibull`
    """
    rs = _generator.get_random_state()
    return rs.weibull(a, size=size, dtype=dtype)