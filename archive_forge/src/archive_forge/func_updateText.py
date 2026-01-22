import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def updateText(self, **kwargs):
    self.skipValidate = True
    txt = self.formatText(**kwargs)
    self.lineEdit().setText(txt)
    self.lastText = txt
    self.skipValidate = False