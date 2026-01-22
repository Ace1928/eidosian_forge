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
def dataRect(self):
    """
        Returns a bounding rectangle (as :class:`QtCore.QRectF`) for the full set of data.
        Will return `None` if there is no data or if all values (x or y) are NaN.
        """
    if self._dataset is None:
        return None
    return self._dataset.dataRect()