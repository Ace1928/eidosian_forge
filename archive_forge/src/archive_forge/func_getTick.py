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
def getTick(self, tick):
    """Return the Tick object at the specified index.
        
        ==============  ==================================================================
        **Arguments:**
        tick            An integer corresponding to the index of the desired tick. If the
                        argument is not an integer it will be returned unchanged.
        ==============  ==================================================================
        """
    if type(tick) is int:
        tick = self.listTicks()[tick][0]
    return tick