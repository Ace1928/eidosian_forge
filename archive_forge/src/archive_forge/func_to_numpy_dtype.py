from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
def to_numpy_dtype(self):
    """
        To Numpy record dtype.
        """
    return np.dtype([('f%d' % i, to_numpy_dtype(typ)) for i, typ in enumerate(self.parameters[0])])