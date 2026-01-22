import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def makeCrosshair(r=0.5, w=1, h=1):
    path = QtGui.QPainterPath()
    rect = QtCore.QRectF(-r, -r, r * 2, r * 2)
    path.addEllipse(rect)
    path.moveTo(-w, 0)
    path.lineTo(w, 0)
    path.moveTo(0, -h)
    path.lineTo(0, h)
    return path