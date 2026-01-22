import numpy as np
from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def setLookupTable(self, lut, update=True):
    self.cmap = None
    self.lut_qcolor = lut[:]
    if update:
        self._rerender(autoLevels=False)
        self.update()