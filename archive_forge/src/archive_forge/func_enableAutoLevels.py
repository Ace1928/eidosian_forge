import numpy as np
from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def enableAutoLevels(self):
    self._defaultAutoLevels = True