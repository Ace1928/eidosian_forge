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
def itemBoundsChanged(self, item):
    self._itemBoundsCache.pop(item, None)
    if self.state['autoRange'][0] is not False or self.state['autoRange'][1] is not False:
        self._autoRangeNeedsUpdate = True
        self.update()