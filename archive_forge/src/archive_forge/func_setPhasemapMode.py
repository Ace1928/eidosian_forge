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
def setPhasemapMode(self, state):
    """
        ``state = True`` enables phase map mode, where a mapping 
        according to ``x_mappped = y`` and ``y_mapped = dy / dx``
        is applied, plotting the numerical derivative of the data over the 
        original `y` values.
        """
    if self.opts['phasemapMode'] == state:
        return
    self.opts['phasemapMode'] = state
    self._datasetMapped = None
    self._datasetDisplay = None
    self._adsLastValue = 1
    self.updateItems(styleUpdate=False)
    self.informViewBoundsChanged()