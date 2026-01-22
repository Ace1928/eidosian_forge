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
def traceImage(image, values, smooth=0.5):
    """
    Convert an image to a set of QPainterPath curves.
    One curve will be generated for each item in *values*; each curve outlines the area
    of the image that is closer to its value than to any others.
    
    If image is RGB or RGBA, then the shape of values should be (nvals, 3/4)
    The parameter *smooth* is expressed in pixels.
    """
    if values.ndim == 2:
        values = values.T
    values = values[np.newaxis, np.newaxis, ...].astype(float)
    image = image[..., np.newaxis].astype(float)
    diff = np.abs(image - values)
    if values.ndim == 4:
        diff = diff.sum(axis=2)
    labels = np.argmin(diff, axis=2)
    paths = []
    for i in range(diff.shape[-1]):
        d = (labels == i).astype(float)
        d = gaussianFilter(d, (smooth, smooth))
        lines = isocurve(d, 0.5, connected=True, extendToEdge=True)
        path = QtGui.QPainterPath()
        for line in lines:
            path.moveTo(*line[0])
            for p in line[1:]:
                path.lineTo(*p)
        paths.append(path)
    return paths