import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def dependentNodes(self):
    """Return the list of nodes which receive input from this terminal."""
    return set([t.node() for t in self.connections() if t.isInput()])