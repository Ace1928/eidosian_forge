import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def setWrapping(self, s):
    """Set whether spin box is circular.
        
        Both bounds must be set for this to have an effect."""
    self.opts['wrapping'] = s