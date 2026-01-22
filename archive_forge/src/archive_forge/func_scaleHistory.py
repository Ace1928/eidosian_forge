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
def scaleHistory(self, d):
    if len(self.axHistory) == 0:
        return
    ptr = max(0, min(len(self.axHistory) - 1, self.axHistoryPointer + d))
    if ptr != self.axHistoryPointer:
        self.axHistoryPointer = ptr
        self.showAxRect(self.axHistory[ptr])