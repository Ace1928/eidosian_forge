from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def setClickable(self, s, width=None):
    """Sets whether the item responds to mouse clicks.

        The `width` argument specifies the width in pixels orthogonal to the
        curve that will respond to a mouse click.
        """
    self.clickable = s
    if width is not None:
        self.opts['mouseWidth'] = width
        self._mouseShape = None
        self._boundingRect = None