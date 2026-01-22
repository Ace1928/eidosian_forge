import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def setRange(self, minValue, maxValue):
    if minValue < 0 or maxValue > 99 or minValue > maxValue:
        QtCore.qWarning('LCDRange::setRange(%d, %d)\n\tRange must be 0..99\n\tand minValue must not be greater than maxValue' % (minValue, maxValue))
        return
    self.slider.setRange(minValue, maxValue)