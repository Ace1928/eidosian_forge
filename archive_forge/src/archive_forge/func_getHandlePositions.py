import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def getHandlePositions(self):
    """Return the positions of all handles in local coordinates."""
    pos = [self.mapFromScene(self.lines[0].getHandles()[0].scenePos())]
    for l in self.lines:
        pos.append(self.mapFromScene(l.getHandles()[1].scenePos()))
    return pos