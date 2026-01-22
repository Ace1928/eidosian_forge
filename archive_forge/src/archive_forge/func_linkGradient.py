import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
def linkGradient(self, slaveGradient, connect=True):
    if connect:
        fn = lambda g, slave=slaveGradient: slave.restoreState(g.saveState())
        self.linkedGradients[id(slaveGradient)] = fn
        self.sigGradientChanged.connect(fn)
        self.sigGradientChanged.emit(self)
    else:
        fn = self.linkedGradients.get(id(slaveGradient), None)
        if fn:
            self.sigGradientChanged.disconnect(fn)