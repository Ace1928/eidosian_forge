from functools import reduce
from operator import mul
from typing import List, Tuple
def mul_shapes(lh_shape: Tuple[int, ...], rh_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Give the shape resulting from multiplying two shapes.

    Adheres the semantics of np.matmul and additionally permits multiplication
    by scalars.

    Parameters
    ----------
    lh_shape : tuple
        The left-hand shape of a multiplication operation.
    rh_shape :  tuple
        The right-hand shape of a multiplication operation.

    Returns
    -------
    tuple
        The shape of the product as per matmul semantics.

    Raises
    ------
    ValueError
        If either of the shapes are scalar.
    """
    lh_old = lh_shape
    rh_old = rh_shape
    lh_shape, rh_shape, shape = mul_shapes_promote(lh_shape, rh_shape)
    if lh_shape != lh_old:
        shape = shape[1:]
    if rh_shape != rh_old:
        shape = shape[:-1]
    return shape