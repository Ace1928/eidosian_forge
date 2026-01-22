import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
def mask_indices(n, mask_func, k=0):
    """
    Return the indices to access (n, n) arrays, given a masking function.

    Assume `mask_func` is a function that, for a square array a of
    size ``(n, n)`` with a possible offset argument `k`, when called
    as ``mask_func(a, k)`` returns a new array with zeros in certain
    locations (functions like :func:`~cupy.triu` or :func:`~cupy.tril` do
    precisely this). Then this function returns the indices where the non-zero
    values would be located.

    Args:
        n (int): The returned indices will be valid to access arrays
            of shape (n, n).
        mask_func (callable): A function whose call signature is
            similar to that of :func:`~cupy.triu`, :func:`~tril`.  That is,
            ``mask_func(x, k)`` returns a boolean array, shaped like
            `x`.  `k` is an optional argument to the function.
        k (scalar): An optional argument which is passed through to
            `mask_func`. Functions like :func:`~cupy.triu`, :func:`~cupy.tril`
            take a second argument that is interpreted as an offset.

    Returns:
        tuple of arrays: The `n` arrays of indices corresponding to
        the locations where ``mask_func(np.ones((n, n)), k)`` is
        True.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.mask_indices`
    """
    a = cupy.ones((n, n), dtype=cupy.int8)
    return mask_func(a, k).nonzero()