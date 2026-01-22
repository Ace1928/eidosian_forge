import sys
from collections import OrderedDict
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal
def setTerminalOffset(self, new_offset):
    """
        This method sets the rendering offset introduced after every terminal of the node.
        This method automatically updates the terminal labels. The default for this value is 12px.

        :param new_offset: The new offset to use in pixels at 100% scale.
        """
    self._nodeOffset = new_offset
    self.updateTerminals()