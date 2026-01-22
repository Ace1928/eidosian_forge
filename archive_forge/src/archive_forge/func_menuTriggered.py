import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def menuTriggered(self, action):
    name, source = action.data()
    if name is None:
        cmap = None
    elif source == 'preset-gradient':
        cmap = preset_gradient_to_colormap(name)
    else:
        cmap = colormap.get(name, source=source)
    self.setColorMap(cmap)