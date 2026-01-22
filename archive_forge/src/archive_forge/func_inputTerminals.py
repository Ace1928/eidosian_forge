import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def inputTerminals(self):
    """Return the terminal(s) that give input to this one."""
    return [t for t in self.connections() if t.isOutput()]