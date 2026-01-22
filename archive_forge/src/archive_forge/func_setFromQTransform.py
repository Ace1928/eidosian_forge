from math import atan2, degrees
import numpy as np
from . import SRTTransform3D
from .Point import Point
from .Qt import QtGui
def setFromQTransform(self, tr):
    p1 = Point(tr.map(0.0, 0.0))
    p2 = Point(tr.map(1.0, 0.0))
    p3 = Point(tr.map(0.0, 1.0))
    dp2 = Point(p2 - p1)
    dp3 = Point(p3 - p1)
    if dp2.angle(dp3, units='radians') > 0:
        da = 0
        sy = -1.0
    else:
        da = 0
        sy = 1.0
    self._state = {'pos': Point(p1), 'scale': Point(dp2.length(), dp3.length() * sy), 'angle': degrees(atan2(dp2[1], dp2[0])) + da}
    self.update()