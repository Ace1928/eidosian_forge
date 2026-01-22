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
def setLeftButtonAction(self, mode='rect'):
    if mode.lower() == 'rect':
        self.setMouseMode(ViewBox.RectMode)
    elif mode.lower() == 'pan':
        self.setMouseMode(ViewBox.PanMode)
    else:
        raise Exception('graphicsItems:ViewBox:setLeftButtonAction: unknown mode = %s (Options are "pan" and "rect")' % mode)