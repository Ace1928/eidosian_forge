import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def check_reshape_kwargs(kwargs):
    """Unpack keyword arguments for reshape function.

    This is useful because keyword arguments after star arguments are not
    allowed in Python 2, but star keyword arguments are. This function unpacks
    'order' and 'copy' from the star keyword arguments (with defaults) and
    throws an error for any remaining.
    """
    order = kwargs.pop('order', 'C')
    copy = kwargs.pop('copy', False)
    if kwargs:
        raise TypeError('reshape() got unexpected keywords arguments: {}'.format(', '.join(kwargs.keys())))
    return (order, copy)