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
def setSymbol(self, symbol):
    """ `symbol` can be any string recognized by 
        :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` or a list that
        specifies a symbol for each point.
        """
    if self.opts['symbol'] == symbol:
        return
    self.opts['symbol'] = symbol
    self.updateItems(styleUpdate=True)