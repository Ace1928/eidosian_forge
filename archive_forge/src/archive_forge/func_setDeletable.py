import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def setDeletable(self, b):
    self.deletable = b
    if b:
        self.setAcceptedMouseButtons(self.acceptedMouseButtons() | QtCore.Qt.MouseButton.RightButton)
    else:
        self.setAcceptedMouseButtons(self.acceptedMouseButtons() & ~QtCore.Qt.MouseButton.RightButton)