import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def movePoint(self, pos, modifiers=None, finish=True):
    if modifiers is None:
        modifiers = QtCore.Qt.KeyboardModifier.NoModifier
    for r in self.rois:
        if not r.checkPointMove(self, pos, modifiers):
            return
    for r in self.rois:
        r.movePoint(self, pos, modifiers, finish=finish, coords='scene')