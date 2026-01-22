import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def labelString(self):
    if self.labelUnits == '':
        if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
            units = ''
        else:
            units = '(x%g)' % (1.0 / self.autoSIPrefixScale)
    else:
        units = '(%s%s)' % (self.labelUnitPrefix, self.labelUnits)
    s = '%s %s' % (self.labelText, units)
    style = ';'.join(['%s: %s' % (k, self.labelStyle[k]) for k in self.labelStyle])
    return "<span style='%s'>%s</span>" % (style, s)