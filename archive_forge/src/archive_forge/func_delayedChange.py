import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def delayedChange(self):
    try:
        if not fn.eq(self.val, self.lastValEmitted):
            self.emitChanged()
    except RuntimeError:
        pass