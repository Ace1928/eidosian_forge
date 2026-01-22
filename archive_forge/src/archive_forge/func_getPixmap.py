import warnings
from collections.abc import Callable
import numpy
from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject
def getPixmap(self):
    if self._renderRequired:
        self.render()
        if self._unrenderable:
            return None
    return QtGui.QPixmap.fromImage(self.qimage)