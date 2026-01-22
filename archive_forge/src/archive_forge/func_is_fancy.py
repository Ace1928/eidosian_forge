import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def is_fancy(sliceobj):
    """Returns True if sliceobj is attempting fancy indexing

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``

    Returns
    -------
    tf: bool
        True if sliceobj represents fancy indexing, False for basic indexing
    """
    if not isinstance(sliceobj, tuple):
        sliceobj = (sliceobj,)
    for slicer in sliceobj:
        if getattr(slicer, 'ndim', 0) > 0:
            return True
        if isinstance(slicer, slice) or slicer in (None, Ellipsis):
            continue
        try:
            int(slicer)
        except TypeError:
            return True
    return False