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
def mapFromItemToView(self, item, obj):
    """Maps *obj* from the local coordinate system of *item* to the view coordinates"""
    self.updateMatrix()
    return self.childGroup.mapFromItem(item, obj)