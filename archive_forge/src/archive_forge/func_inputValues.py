import sys
from collections import OrderedDict
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal
def inputValues(self):
    """Return a dict of all input values currently assigned to this node."""
    vals = {}
    for n, t in self.inputs().items():
        vals[n] = t.value()
    return vals