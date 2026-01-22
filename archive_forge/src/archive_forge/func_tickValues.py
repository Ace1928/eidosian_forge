import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def tickValues(self, minVal, maxVal, size):
    """
        Return the values and spacing of ticks to draw::

            [
                (spacing, [major ticks]),
                (spacing, [minor ticks]),
                ...
            ]

        By default, this method calls tickSpacing to determine the correct tick locations.
        This is a good method to override in subclasses.
        """
    minVal, maxVal = sorted((minVal, maxVal))
    minVal *= self.scale
    maxVal *= self.scale
    ticks = []
    tickLevels = self.tickSpacing(minVal, maxVal, size)
    allValues = np.array([])
    for i in range(len(tickLevels)):
        spacing, offset = tickLevels[i]
        start = ceil((minVal - offset) / spacing) * spacing + offset
        num = int((maxVal - start) / spacing) + 1
        values = (np.arange(num) * spacing + start) / self.scale
        close = np.any(np.isclose(allValues, values[:, np.newaxis], rtol=0, atol=spacing / self.scale * 0.01), axis=-1)
        values = values[~close]
        allValues = np.concatenate([allValues, values])
        ticks.append((spacing / self.scale, values.tolist()))
    if self.logMode:
        return self.logTickValues(minVal, maxVal, size, ticks)
    return ticks