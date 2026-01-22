import importlib
import os
from collections import OrderedDict
from numpy import ndarray
from .. import DataTreeWidget, FileDialog
from .. import configfile as configfile
from .. import dockarea as dockarea
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtWidgets
from . import FlowchartCtrlTemplate_generic as FlowchartCtrlTemplate
from . import FlowchartGraphicsView
from .library import LIBRARY
from .Node import Node
from .Terminal import Terminal
def createNode(self, nodeType, name=None, pos=None):
    """Create a new Node and add it to this flowchart.
        """
    if name is None:
        n = 0
        while True:
            name = '%s.%d' % (nodeType, n)
            if name not in self._nodes:
                break
            n += 1
    node = self.library.getNodeType(nodeType)(name)
    self.addNode(node, name, pos)
    return node