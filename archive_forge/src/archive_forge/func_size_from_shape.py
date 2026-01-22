from functools import reduce
from operator import mul
from typing import List, Tuple
def size_from_shape(shape) -> int:
    """ Compute the size of a given shape by multiplying the sizes of each axis.

    This is a replacement for np.prod(shape, dtype=int) which is much slower for
    small arrays than the implementation below.

    Parameters
    ----------
    shape : tuple
        a tuple of integers describing the shape of an object

    Returns
    -------
    int
        The size of an object corresponding to shape.
    """
    return reduce(mul, shape, 1)