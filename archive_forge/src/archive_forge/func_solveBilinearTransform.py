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
def solveBilinearTransform(points1, points2):
    """
    Find a bilinear transformation matrix (2x4) that maps points1 onto points2.
    Points must be specified as a list of 4 Vector, Point, QPointF, etc.
    
    To use this matrix to map a point [x,y]::
    
        mapped = np.dot(matrix, [x*y, x, y, 1])
    """
    import numpy.linalg
    A = np.array([[points1[i].x() * points1[i].y(), points1[i].x(), points1[i].y(), 1] for i in range(4)])
    B = np.array([[points2[i].x(), points2[i].y()] for i in range(4)])
    matrix = np.zeros((2, 4))
    for i in range(2):
        matrix[i] = numpy.linalg.solve(A, B[:, i])
    return matrix