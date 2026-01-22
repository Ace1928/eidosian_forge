import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def emitChanged(self):
    self.lastValEmitted = self.val
    self.valueChanged.emit(float(self.val))
    self.sigValueChanged.emit(self)