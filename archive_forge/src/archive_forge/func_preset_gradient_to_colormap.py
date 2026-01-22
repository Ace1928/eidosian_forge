import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def preset_gradient_to_colormap(name):
    if name == 'spectrum':
        cmap = colormap.makeHslCycle((0, 300 / 360), steps=30)
    elif name == 'cyclic':
        cmap = colormap.makeHslCycle((1, 0))
    else:
        cmap = colormap.ColorMap(*zip(*Gradients[name]['ticks']), name=name)
    return cmap