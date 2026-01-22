from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def mapToQColor(self, data):
    """Convenience function; see :func:`map() <pyqtgraph.ColorMap.map>`."""
    return self.map(data, mode=self.QCOLOR)