import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def setMultiValue(self, multi):
    """Set whether this is a multi-value terminal."""
    self._multi = multi
    if not multi and len(self.inputTerminals()) > 1:
        self.disconnectAll()
    for term in self.inputTerminals():
        self.inputChanged(term)