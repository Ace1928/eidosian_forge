import warnings
from ..Qt import QtCore, QtGui, QtWidgets
def setOrientation(self, o):
    if self.orientation == o:
        return
    self.orientation = o
    self.update()
    self.updateGeometry()