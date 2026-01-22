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
def ndarray_from_qpolygonf(polyline):
    vp = Qt.compat.voidptr(polyline.data(), len(polyline) * 2 * 8, True)
    return np.frombuffer(vp, dtype=np.float64).reshape((-1, 2))