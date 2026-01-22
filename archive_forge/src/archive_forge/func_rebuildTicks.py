from .. import functions as fn
from ..Qt import QtGui, QtWidgets
from .UIGraphicsItem import UIGraphicsItem
def rebuildTicks(self):
    self.path = QtGui.QPainterPath()
    for x in self.xvals:
        self.path.moveTo(x, 0.0)
        self.path.lineTo(x, 1.0)