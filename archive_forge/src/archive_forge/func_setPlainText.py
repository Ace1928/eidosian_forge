from math import atan2, degrees
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject
def setPlainText(self, text):
    """
        Set the plain text to be rendered by this item. 
        
        See QtWidgets.QGraphicsTextItem.setPlainText().
        """
    if text != self.toPlainText():
        self.textItem.setPlainText(text)
        self.updateTextPos()