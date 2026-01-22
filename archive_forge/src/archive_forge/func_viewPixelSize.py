import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def viewPixelSize(self):
    """Return the (width, height) of a screen pixel in view coordinates."""
    if self._viewPixelSizeCache is None:
        o = self.mapToView(Point(0, 0))
        px, py = [Point(self.mapToView(v) - o) for v in self.pixelVectors()]
        self._viewPixelSizeCache = (px.length(), py.length())
    return self._viewPixelSizeCache