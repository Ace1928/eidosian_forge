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
def raiseColorDialog(self, tick):
    if not tick.colorChangeAllowed:
        return
    self.currentTick = tick
    self.currentTickColor = tick.color
    self.colorDialog.setCurrentColor(tick.color)
    self.colorDialog.open()