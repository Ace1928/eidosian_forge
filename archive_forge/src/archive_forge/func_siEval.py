from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def siEval(s, typ=float, regex=FLOAT_REGEX, suffix=None):
    """
    Convert a value written in SI notation to its equivalent prefixless value.

    Example::
    
        siEval("100 μV")  # returns 0.0001
    """
    val, siprefix, suffix = siParse(s, regex, suffix=suffix)
    v = typ(val)
    return siApply(v, siprefix)