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
def ndarray_to_qimage(arr, fmt):
    """
    Low level function to encapsulate QImage creation differences between bindings.
    "arr" is assumed to be C-contiguous. 
    """
    if QT_LIB.startswith('PyQt'):
        img_ptr = int(Qt.sip.voidptr(arr))
    else:
        img_ptr = arr
    h, w = arr.shape[:2]
    bytesPerLine = arr.strides[0]
    qimg = QtGui.QImage(img_ptr, w, h, bytesPerLine, fmt)
    qimg.data = arr
    return qimg