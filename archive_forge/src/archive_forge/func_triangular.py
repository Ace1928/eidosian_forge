from cupy.random import _generator
from cupy import _util
def triangular(left, mode, right, size=None, dtype=float):
    """Triangular distribution.

    Returns an array of samples drawn from the triangular distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\begin{cases}
            \\frac{2(x-l)}{(r-l)(m-l)} & \\text{for } l \\leq x \\leq m, \\\\
            \\frac{2(r-x)}{(r-l)(r-m)} & \\text{for } m \\leq x \\leq r, \\\\
            0 & \\text{otherwise}.
          \\end{cases}

    Args:
        left (float): Lower limit :math:`l`.
        mode (float): The value where the peak of the distribution occurs.
            :math:`m`.
        right (float): Higher Limit :math:`r`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the triangular distribution.

    .. seealso::
        :func:`numpy.random.triangular`
    """
    rs = _generator.get_random_state()
    return rs.triangular(left, mode, right, size, dtype)