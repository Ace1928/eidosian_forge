from math import atan2, degrees
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject
def setTextWidth(self, *args):
    """
        Set the width of the text.
        
        If the text requires more space than the width limit, then it will be
        wrapped into multiple lines.
        
        See QtWidgets.QGraphicsTextItem.setTextWidth().
        """
    self.textItem.setTextWidth(*args)
    self.updateTextPos()