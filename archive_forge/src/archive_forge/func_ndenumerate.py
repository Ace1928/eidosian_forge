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
def ndenumerate(a, compressed=True):
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values,
    skipping elements that are masked. With `compressed=False`,
    `ma.masked` is yielded as the value of masked elements. This
    behavior differs from that of `numpy.ndenumerate`, which yields the
    value of the underlying data array.

    Notes
    -----
    .. versionadded:: 1.23.0

    Parameters
    ----------
    a : array_like
        An array with (possibly) masked elements.
    compressed : bool, optional
        If True (default), masked elements are skipped.

    See Also
    --------
    numpy.ndenumerate : Equivalent function ignoring any mask.

    Examples
    --------
    >>> a = np.ma.arange(9).reshape((3, 3))
    >>> a[1, 0] = np.ma.masked
    >>> a[1, 2] = np.ma.masked
    >>> a[2, 1] = np.ma.masked
    >>> a
    masked_array(
      data=[[0, 1, 2],
            [--, 4, --],
            [6, --, 8]],
      mask=[[False, False, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=999999)
    >>> for index, x in np.ma.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 1) 4
    (2, 0) 6
    (2, 2) 8

    >>> for index, x in np.ma.ndenumerate(a, compressed=False):
    ...     print(index, x)
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) --
    (1, 1) 4
    (1, 2) --
    (2, 0) 6
    (2, 1) --
    (2, 2) 8
    """
    for it, mask in zip(np.ndenumerate(a), getmaskarray(a).flat):
        if not mask:
            yield it
        elif not compressed:
            yield (it[0], masked)