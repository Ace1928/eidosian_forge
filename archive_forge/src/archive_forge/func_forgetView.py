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
@staticmethod
def forgetView(vid, name):
    if ViewBox is None:
        return
    if QtWidgets.QApplication.instance() is None:
        return
    for v in list(ViewBox.AllViews.keys()):
        if id(v) == vid:
            ViewBox.AllViews.pop(v)
            break
    ViewBox.NamedViews.pop(name, None)
    ViewBox.updateAllViewLists()