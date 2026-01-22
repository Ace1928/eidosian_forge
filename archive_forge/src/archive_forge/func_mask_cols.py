import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
def mask_cols(a, axis=np._NoValue):
    """
    Mask columns of a 2D array that contain masked values.

    This function is a shortcut to ``mask_rowcols`` with `axis` equal to 1.

    See Also
    --------
    mask_rowcols : Mask rows and/or columns of a 2D array.
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)
    >>> ma.mask_cols(a)
    masked_array(
      data=[[0, --, 0],
            [0, --, 0],
            [0, --, 0]],
      mask=[[False,  True, False],
            [False,  True, False],
            [False,  True, False]],
      fill_value=1)

    """
    if axis is not np._NoValue:
        warnings.warn('The axis argument has always been ignored, in future passing it will raise TypeError', DeprecationWarning, stacklevel=2)
    return mask_rowcols(a, 1)