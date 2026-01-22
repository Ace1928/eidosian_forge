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
def setDefaultPadding(self, padding=0.02):
    """
        Sets the fraction of the data range that is used to pad the view range in when auto-ranging.
        By default, this fraction is 0.02.
        """
    self.state['defaultPadding'] = padding