import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def wrapping(self):
    """Return whether or not the spin box is circular."""
    return self.opts['wrapping']