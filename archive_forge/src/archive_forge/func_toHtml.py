from math import atan2, degrees
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject
def toHtml(self):
    return self.textItem.toHtml()