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
def internalTerminal(self, term):
    """If the terminal belongs to the external Node, return the corresponding internal terminal"""
    if term.node() is self:
        if term.isInput():
            return self.inputNode[term.name()]
        else:
            return self.outputNode[term.name()]
    else:
        return term