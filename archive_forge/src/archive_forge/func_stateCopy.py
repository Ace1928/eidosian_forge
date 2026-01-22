import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def stateCopy(self):
    sc = {}
    sc['pos'] = Point(self.state['pos'])
    sc['size'] = Point(self.state['size'])
    sc['angle'] = self.state['angle']
    return sc