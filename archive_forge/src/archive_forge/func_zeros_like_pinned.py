import numpy
from cupy import cuda
from cupy._creation.basic import _new_like_order_and_strides
from cupy._core import internal
def zeros_like_pinned(a, dtype=None, order='K', subok=None, shape=None):
    """Returns a new, zero-initialized NumPy array with the same shape and dtype
    as those of the given array.

    This is a convenience function which is just :func:`numpy.zeros_like`,
    except that the underlying memory is pinned/pagelocked.

    This function currently does not support ``subok`` option.

    Args:
        a (numpy.ndarray or cupy.ndarray): Base array.
        dtype: Data type specifier. The dtype of ``a`` is used by default.
        order ({'C', 'F', 'A', or 'K'}): Overrides the memory layout of the
            result. ``'C'`` means C-order, ``'F'`` means F-order, ``'A'`` means
            ``'F'`` if ``a`` is Fortran contiguous, ``'C'`` otherwise.
            ``'K'`` means match the layout of ``a`` as closely as possible.
        subok: Not supported yet, must be None.
        shape (int or tuple of ints): Overrides the shape of the result. If
            ``order='K'`` and the number of dimensions is unchanged, will try
            to keep order, otherwise, ``order='C'`` is implied.

    Returns:
        numpy.ndarray: An array filled with zeros.

    .. seealso:: :func:`numpy.zeros_like`

    """
    out = empty_like_pinned(a, dtype, order, subok, shape)
    numpy.copyto(out, 0, casting='unsafe')
    return out