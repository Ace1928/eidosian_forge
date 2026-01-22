import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def getSnapPosition(self, pos, snap=None):
    if snap is None or snap is True:
        if self.snapSize is None:
            return pos
        snap = Point(self.snapSize, self.snapSize)
    return Point(round(pos[0] / snap[0]) * snap[0], round(pos[1] / snap[1]) * snap[1])