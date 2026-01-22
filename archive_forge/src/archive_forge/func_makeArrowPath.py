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
def makeArrowPath(headLen=20, headWidth=None, tipAngle=20, tailLen=20, tailWidth=3, baseAngle=0):
    """
    Construct a path outlining an arrow with the given dimensions.
    The arrow points in the -x direction with tip positioned at 0,0.
    If *headWidth* is supplied, it overrides *tipAngle* (in degrees).
    If *tailLen* is None, no tail will be drawn.
    """
    if headWidth is None:
        headWidth = headLen * math.tan(math.radians(tipAngle * 0.5))
    path = QtGui.QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(headLen, -headWidth)
    if tailLen is None:
        innerY = headLen - headWidth * math.tan(math.radians(baseAngle))
        path.lineTo(innerY, 0)
    else:
        tailWidth *= 0.5
        innerY = headLen - (headWidth - tailWidth) * math.tan(math.radians(baseAngle))
        path.lineTo(innerY, -tailWidth)
        path.lineTo(headLen + tailLen, -tailWidth)
        path.lineTo(headLen + tailLen, tailWidth)
        path.lineTo(innerY, tailWidth)
    path.lineTo(headLen, headWidth)
    path.lineTo(0, 0)
    return path