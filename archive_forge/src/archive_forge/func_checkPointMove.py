import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def checkPointMove(self, handle, pos, modifiers):
    """When handles move, they must ask the ROI if the move is acceptable.
        By default, this always returns True. Subclasses may wish override.
        """
    return True