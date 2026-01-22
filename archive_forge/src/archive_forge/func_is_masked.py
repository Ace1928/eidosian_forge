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
def is_masked(x):
    """
    Determine whether input has masked values.

    Accepts any object as input, but always returns False unless the
    input is a MaskedArray containing masked values.

    Parameters
    ----------
    x : array_like
        Array to check for masked values.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray with masked values, False otherwise.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 0)
    >>> x
    masked_array(data=[--, 1, --, 2, 3],
                 mask=[ True, False,  True, False, False],
           fill_value=0)
    >>> ma.is_masked(x)
    True
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 42)
    >>> x
    masked_array(data=[0, 1, 0, 2, 3],
                 mask=False,
           fill_value=42)
    >>> ma.is_masked(x)
    False

    Always returns False if `x` isn't a MaskedArray.

    >>> x = [False, True, False]
    >>> ma.is_masked(x)
    False
    >>> x = 'a string'
    >>> ma.is_masked(x)
    False

    """
    m = getmask(x)
    if m is nomask:
        return False
    elif m.any():
        return True
    return False