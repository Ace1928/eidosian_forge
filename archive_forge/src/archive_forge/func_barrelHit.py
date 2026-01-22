import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
def barrelHit(self, pos):
    matrix = QtGui.QMatrix()
    matrix.translate(0, self.height())
    matrix.rotate(-self.currentAngle)
    matrix, invertible = matrix.inverted()
    return self.barrelRect.contains(matrix.map(pos))