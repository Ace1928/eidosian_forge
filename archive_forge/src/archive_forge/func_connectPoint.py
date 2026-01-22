import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def connectPoint(self):
    return self.mapToView(self.mapFromItem(self.box, self.box.boundingRect().center()))