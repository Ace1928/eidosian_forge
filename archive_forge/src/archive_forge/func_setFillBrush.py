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
def setFillBrush(self, *args, **kargs):
    """ 
        Sets the :class:`QtGui.QBrush` used to fill the area under the curve.
        See :func:`mkBrush() <pyqtgraph.mkBrush>`) for arguments.
        """
    if args and args[0] is None:
        brush = None
    else:
        brush = fn.mkBrush(*args, **kargs)
    if self.opts['fillBrush'] == brush:
        return
    self.opts['fillBrush'] = brush
    self.updateItems(styleUpdate=True)