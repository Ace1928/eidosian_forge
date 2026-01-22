from functools import reduce
from operator import mul
from typing import List, Tuple
def mul_shapes_promote(lh_shape: Tuple[int, ...], rh_shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Promotes shapes as necessary and returns promoted shape of product.

    If lh_shape is of length one, prepend a one to it.
    If rh_shape is of length one, append a one to it.

    Parameters
    ----------
    lh_shape : tuple
        The left-hand shape of a multiplication operation.
    rh_shape : tuple
        The right-hand shape of a multiplication operation.

    Returns
    -------
    tuple
        The promoted left-hand shape.
    tuple
        The promoted right-hand shape.
    tuple
        The promoted shape of the product.

    Raises
    ------
    ValueError
        If either of the shapes are 0D.
    """
    if not lh_shape or not rh_shape:
        raise ValueError('Multiplication by scalars is not permitted.')
    if len(lh_shape) == 1:
        lh_shape = (1,) + lh_shape
    if len(rh_shape) == 1:
        rh_shape = rh_shape + (1,)
    lh_mat_shape = lh_shape[-2:]
    rh_mat_shape = rh_shape[-2:]
    if lh_mat_shape[1] != rh_mat_shape[0]:
        raise ValueError('Incompatible dimensions %s %s' % (lh_shape, rh_shape))
    if lh_shape[:-2] != rh_shape[:-2]:
        raise ValueError('Incompatible dimensions %s %s' % (lh_shape, rh_shape))
    return (lh_shape, rh_shape, tuple(list(lh_shape[:-2]) + [lh_mat_shape[0]] + [rh_mat_shape[1]]))