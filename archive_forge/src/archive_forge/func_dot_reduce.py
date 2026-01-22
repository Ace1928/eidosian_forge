from functools import reduce
import numpy as np
def dot_reduce(*args):
    """Apply numpy dot product function from right to left on arrays

    For passed arrays :math:`A, B, C, ... Z` returns :math:`A \\dot B \\dot C ...
    \\dot Z` where "." is the numpy array dot product.

    Parameters
    ----------
    \\*\\*args : arrays
        Arrays that can be passed to numpy ``dot`` function

    Returns
    -------
    dot_product : array
        If there are N arguments, result of ``arg[0].dot(arg[1].dot(arg[2].dot
        ...  arg[N-2].dot(arg[N-1])))...``
    """
    return reduce(lambda x, y: np.dot(y, x), args[::-1])