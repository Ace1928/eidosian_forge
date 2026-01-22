import cupy
def zero_ext(x, n, axis=-1):
    """
    Zero padding at the boundaries of an array

    Generate a new ndarray that is a zero-padded extension of `x` along
    an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the
        axis.
    axis : int, optional
        The axis along which to extend `x`. Default is -1.

    Examples
    --------
    >>> from cupyx.scipy.signal._arraytools import zero_ext
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> zero_ext(a, 2)
    array([[ 0,  0,  1,  2,  3,  4,  5,  0,  0],
           [ 0,  0,  0,  1,  4,  9, 16,  0,  0]])
    """
    if n < 1:
        return x
    zeros_shape = list(x.shape)
    zeros_shape[axis] = n
    zeros = cupy.zeros(zeros_shape, dtype=x.dtype)
    ext = cupy.concatenate((zeros, x, zeros), axis=axis)
    return ext