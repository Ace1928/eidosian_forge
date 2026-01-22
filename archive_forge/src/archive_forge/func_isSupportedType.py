import sys
from PySide2 import QtCore, QtGui, QtWidgets
@staticmethod
def isSupportedType(value):
    return isinstance(value, (bool, float, int, QtCore.QByteArray, str, QtGui.QColor, QtCore.QDate, QtCore.QDateTime, QtCore.QTime, QtCore.QPoint, QtCore.QRect, QtCore.QSize, list))