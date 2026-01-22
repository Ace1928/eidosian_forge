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
def usesAlpha(self):
    """Return True if any ticks have an alpha < 255"""
    ticks = self.listTicks()
    for t in ticks:
        if t[0].color.alpha() < 255:
            return True
    return False