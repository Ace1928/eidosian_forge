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
def setXRange(self, min, max, padding=None, update=True):
    """
        Set the visible X range of the view to [*min*, *max*].
        The *padding* argument causes the range to be set larger by the fraction specified.
        (by default, this value is between the default padding and 0.1 depending on the size of the ViewBox)
        """
    self.setRange(xRange=[min, max], update=update, padding=padding)