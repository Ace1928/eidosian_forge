import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def updateAutoSIPrefix(self):
    if self.label.isVisible():
        if self.logMode:
            _range = 10 ** np.array(self.range)
        else:
            _range = self.range
        scale, prefix = fn.siScale(max(abs(_range[0] * self.scale), abs(_range[1] * self.scale)))
        if self.labelUnits == '' and prefix in ['k', 'm']:
            scale = 1.0
            prefix = ''
        self.autoSIPrefixScale = scale
        self.labelUnitPrefix = prefix
    else:
        self.autoSIPrefixScale = 1.0
    self._updateLabel()