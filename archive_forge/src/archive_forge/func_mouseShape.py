from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def mouseShape(self):
    """Return the shape of this item after expanding by 2 pixels"""
    shape = self.shape()
    ds = self.mapToDevice(shape)
    stroker = QtGui.QPainterPathStroker()
    stroker.setWidh(2)
    ds2 = stroker.createStroke(ds).united(ds)
    return self.mapFromDevice(ds2)