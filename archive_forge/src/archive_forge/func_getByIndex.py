from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def getByIndex(self, idx):
    """Retrieve a QColor by the index of the stop it is assigned to."""
    return QtGui.QColor.fromRgbF(*self.color[idx])