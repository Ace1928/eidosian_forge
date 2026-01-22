from cupy.random import _generator
from cupy import _util
def zipf(a, size=None, dtype=int):
    """Zipf distribution.

    Returns an array of samples drawn from the Zipf distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = \\frac{x^{-a}}{ \\zeta (a)},

    where :math:`\\zeta` is the Riemann Zeta function.

    Args:
        a (float): Parameter of the beta distribution :math:`a`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the Zipf distribution.

    .. seealso::
        :func:`numpy.random.zipf`
    """
    rs = _generator.get_random_state()
    return rs.zipf(a, size=size, dtype=dtype)