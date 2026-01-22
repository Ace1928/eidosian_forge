import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def setTickPen(self, *args, **kwargs):
    """
        Set the pen used for drawing tick marks.
        If no arguments are given, the default pen will be used.
        """
    self.picture = None
    if args or kwargs:
        self._tickPen = fn.mkPen(*args, **kwargs)
    else:
        self._tickPen = None
    self._updateLabel()