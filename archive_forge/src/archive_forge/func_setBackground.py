from .. import functions as fn
from .. import getConfigOption
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
def setBackground(self, background):
    """
        Set the background color of the GraphicsView.
        To use the defaults specified py pyqtgraph.setConfigOption, use background='default'.
        To make the background transparent, use background=None.
        """
    self._background = background
    if background == 'default':
        background = getConfigOption('background')
    brush = fn.mkBrush(background)
    self.setBackgroundBrush(brush)