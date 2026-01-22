import numpy as np
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .UIGraphicsItem import UIGraphicsItem
def setTextPen(self, *args, **kwargs):
    """Set the pen used to draw the texts."""
    if kwargs == {} and (args == () or args == ('default',)):
        self.opts['textPen'] = fn.mkPen(getConfigOption('foreground'))
    elif args == (None,):
        self.opts['textPen'] = None
    else:
        self.opts['textPen'] = fn.mkPen(*args, **kwargs)
    self.picture = None
    self.update()