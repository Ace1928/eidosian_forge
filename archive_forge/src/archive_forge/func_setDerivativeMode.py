import math
import warnings
import bisect
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem
def setDerivativeMode(self, state):
    """
        ``state = True`` enables derivative mode, where a mapping according to
        ``y_mapped = dy / dx`` is applied, with `dx` and `dy` representing the 
        differences between adjacent `x` and `y` values.
        """
    if self.opts['derivativeMode'] == state:
        return
    self.opts['derivativeMode'] = state
    self._datasetMapped = None
    self._datasetDisplay = None
    self._adsLastValue = 1
    self.updateItems(styleUpdate=False)
    self.informViewBoundsChanged()