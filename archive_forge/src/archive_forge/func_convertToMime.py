import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def convertToMime(self, mime, data, flav):
    all = QtCore.QByteArray()
    for i in data:
        all += i
    return all