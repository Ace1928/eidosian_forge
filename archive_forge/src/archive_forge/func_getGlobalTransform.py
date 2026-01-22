import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def getGlobalTransform(self, relativeTo=None):
    """Return global transformation (rotation angle+translation) required to move 
        from relative state to current state. If relative state isn't specified,
        then we use the state of the ROI when mouse is pressed."""
    if relativeTo is None:
        relativeTo = self.preMoveState
    st = self.getState()
    relativeTo['scale'] = relativeTo['size']
    st['scale'] = st['size']
    t1 = SRTTransform(relativeTo)
    t2 = SRTTransform(st)
    return t2 / t1