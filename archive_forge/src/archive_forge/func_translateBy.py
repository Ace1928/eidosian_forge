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
def translateBy(self, t=None, x=None, y=None):
    """
        Translate the view by *t*, which may be a Point or tuple (x, y).

        Alternately, x or y may be specified independently, leaving the other
        axis unchanged (note that using a translation of 0 may still cause
        small changes due to floating-point error).
        """
    vr = self.targetRect()
    if t is not None:
        t = Point(t)
        self.setRange(vr.translated(t), padding=0)
    else:
        if x is not None:
            x = (vr.left() + x, vr.right() + x)
        if y is not None:
            y = (vr.top() + y, vr.bottom() + y)
        if x is not None or y is not None:
            self.setRange(xRange=x, yRange=y, padding=0)