from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def isMapTrivial(self):
    """
        Returns `True` if the gradient has exactly two stops in it: Black at 0.0 and white at 1.0.
        """
    if len(self.pos) != 2:
        return False
    if self.pos[0] != 0.0 or self.pos[1] != 1.0:
        return False
    if self.color.dtype.kind == 'f':
        return np.all(self.color == np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))
    else:
        return np.all(self.color == np.array([[0, 0, 0, 255], [255, 255, 255, 255]]))