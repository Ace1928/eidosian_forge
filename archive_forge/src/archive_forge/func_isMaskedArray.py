import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def isMaskedArray(x):
    """
    Test whether input is an instance of MaskedArray.

    This function returns True if `x` is an instance of MaskedArray
    and returns False otherwise.  Any object is accepted as input.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray.

    See Also
    --------
    isMA : Alias to isMaskedArray.
    isarray : Alias to isMaskedArray.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.eye(3, 3)
    >>> a
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> m = ma.masked_values(a, 0)
    >>> m
    masked_array(
      data=[[1.0, --, --],
            [--, 1.0, --],
            [--, --, 1.0]],
      mask=[[False,  True,  True],
            [ True, False,  True],
            [ True,  True, False]],
      fill_value=0.0)
    >>> ma.isMaskedArray(a)
    False
    >>> ma.isMaskedArray(m)
    True
    >>> ma.isMaskedArray([0, 1, 2])
    False

    """
    return isinstance(x, MaskedArray)