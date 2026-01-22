import random
from PySide2 import QtCore, QtGui, QtWidgets
def maxY(self):
    m = self.coords[0][1]
    for i in range(4):
        m = max(m, self.coords[i][1])
    return m