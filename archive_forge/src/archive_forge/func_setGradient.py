from .. import functions as fn
from ..Qt import QtCore, QtGui
from .UIGraphicsItem import UIGraphicsItem
def setGradient(self, g):
    self.gradient = g
    self.update()