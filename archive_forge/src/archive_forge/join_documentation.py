import cupy
from cupy import _core
Stacks arrays along a new axis.

    Args:
        tup (sequence of arrays): Arrays to be stacked.
        axis (int): Axis along which the arrays are stacked.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Stacked array.

    .. seealso:: :func:`numpy.stack`
    