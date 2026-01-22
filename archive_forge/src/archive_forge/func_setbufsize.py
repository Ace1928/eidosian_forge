import collections.abc
import contextlib
import contextvars
from .._utils import set_module
from .umath import (
from . import umath
@set_module('numpy')
def setbufsize(size):
    """
    Set the size of the buffer used in ufuncs.

    Parameters
    ----------
    size : int
        Size of buffer.

    """
    if size > 10000000.0:
        raise ValueError('Buffer size, %s, is too big.' % size)
    if size < 5:
        raise ValueError('Buffer size, %s, is too small.' % size)
    if size % 16 != 0:
        raise ValueError('Buffer size, %s, is not a multiple of 16.' % size)
    pyvals = umath.geterrobj()
    old = getbufsize()
    pyvals[0] = size
    umath.seterrobj(pyvals)
    return old