import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def colorMapChanged(self):
    cmap = self.colorMap()
    self.sigColorMapChanged.emit(cmap)
    self.update()