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
def setAutoPan(self, x=None, y=None):
    """Set whether automatic range will only pan (not scale) the view.
        """
    if x is not None:
        self.state['autoPan'][0] = x
    if y is not None:
        self.state['autoPan'][1] = y
    if None not in [x, y]:
        self.updateAutoRange()