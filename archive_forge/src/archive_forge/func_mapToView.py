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
def mapToView(self, obj):
    """Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox"""
    self.updateMatrix()
    m = fn.invertQTransform(self.childTransform())
    return m.map(obj)