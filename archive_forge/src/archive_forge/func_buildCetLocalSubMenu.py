import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def buildCetLocalSubMenu(self):
    source = None
    names = colormap.listMaps(source=source)
    names = [x for x in names if x.startswith('CET')]
    self.buildSubMenu(names, source)