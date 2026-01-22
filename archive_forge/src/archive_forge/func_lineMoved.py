from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def lineMoved(self, i):
    if self.blockLineSignal:
        return
    if self.lines[0].value() > self.lines[1].value():
        if self.swapMode == 'block':
            self.lines[i].setValue(self.lines[1 - i].value())
        elif self.swapMode == 'push':
            self.lines[1 - i].setValue(self.lines[i].value())
    self.prepareGeometryChange()
    self.sigRegionChanged.emit(self)