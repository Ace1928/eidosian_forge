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
def hoverOver(self, items):
    term = None
    for item in items:
        if item is self.hoverItem:
            return
        self.hoverItem = item
        if hasattr(item, 'term') and isinstance(item.term, Terminal):
            term = item.term
            break
    if term is None:
        self.hoverText.setPlainText('')
    else:
        val = term.value()
        if isinstance(val, ndarray):
            val = '%s %s %s' % (type(val).__name__, str(val.shape), str(val.dtype))
        else:
            val = str(val)
            if len(val) > 400:
                val = val[:400] + '...'
        self.hoverText.setPlainText('%s.%s = %s' % (term.node().name(), term.name(), val))