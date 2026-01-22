import numpy
from cupy import _core
def hsplit(ary, indices_or_sections):
    """Splits an array into multiple sub arrays horizontally.

    This is equivalent to ``split`` with ``axis=0`` if ``ary`` has one
    dimension, and otherwise that with ``axis=1``.

    .. seealso:: :func:`cupy.split` for more detail, :func:`numpy.hsplit`

    """
    if ary.ndim == 0:
        raise ValueError('Cannot hsplit a zero-dimensional array')
    if ary.ndim == 1:
        return split(ary, indices_or_sections, 0)
    else:
        return split(ary, indices_or_sections, 1)