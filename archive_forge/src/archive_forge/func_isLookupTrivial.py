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
def isLookupTrivial(self):
    """Return True if the gradient has exactly two stops in it: black at 0.0 and white at 1.0"""
    ticks = self.listTicks()
    if len(ticks) != 2:
        return False
    if ticks[0][1] != 0.0 or ticks[1][1] != 1.0:
        return False
    c1 = ticks[0][0].color.getRgb()
    c2 = ticks[1][0].color.getRgb()
    if c1 != (0, 0, 0, 255) or c2 != (255, 255, 255, 255):
        return False
    return True