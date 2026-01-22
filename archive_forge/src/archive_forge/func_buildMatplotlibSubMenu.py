import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def buildMatplotlibSubMenu(self):
    source = 'matplotlib'
    names = colormap.listMaps(source=source)
    names = [x for x in names if not x.startswith('cet_')]
    names = [x for x in names if not x.endswith('_r')]
    self.buildSubMenu(names, source)