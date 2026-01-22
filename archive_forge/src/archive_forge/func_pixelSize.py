from .. import functions as fn
from .. import getConfigOption
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
def pixelSize(self):
    """Return vector with the length and width of one view pixel in scene coordinates"""
    p0 = Point(0, 0)
    p1 = Point(1, 1)
    tr = self.transform().inverted()[0]
    p01 = tr.map(p0)
    p11 = tr.map(p1)
    return Point(p11 - p01)