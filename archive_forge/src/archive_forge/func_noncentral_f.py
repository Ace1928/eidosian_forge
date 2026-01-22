from cupy.random import _generator
from cupy import _util
def noncentral_f(dfnum, dfden, nonc, size=None, dtype=float):
    """Noncentral F distribution.

    Returns an array of samples drawn from the noncentral F
    distribution.

    Reference: https://en.wikipedia.org/wiki/Noncentral_F-distribution

    Args:
        dfnum (float): Parameter of the noncentral F distribution.
        dfden (float): Parameter of the noncentral F distribution.
        nonc (float): Parameter of the noncentral F distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the noncentral F distribution.

    .. seealso::
        :func:`numpy.random.noncentral_f`
    """
    rs = _generator.get_random_state()
    return rs.noncentral_f(dfnum, dfden, nonc, size=size, dtype=dtype)