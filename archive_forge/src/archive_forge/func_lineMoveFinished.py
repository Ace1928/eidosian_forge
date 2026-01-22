from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def lineMoveFinished(self):
    self.sigRegionChangeFinished.emit(self)