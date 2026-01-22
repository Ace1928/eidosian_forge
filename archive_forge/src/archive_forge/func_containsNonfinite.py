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
@property
def containsNonfinite(self):
    if self.xAllFinite is None or self.yAllFinite is None:
        return None
    return not (self.xAllFinite and self.yAllFinite)