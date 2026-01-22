import sys
from PySide2 import QtCore, QtGui, QtWidgets
@staticmethod
def displayText(value):
    if isinstance(value, (bool, int, QtCore.QByteArray)):
        return str(value)
    if isinstance(value, str):
        return value
    elif isinstance(value, float):
        return '%g' % value
    elif isinstance(value, QtGui.QColor):
        return '(%u,%u,%u,%u)' % (value.red(), value.green(), value.blue(), value.alpha())
    elif isinstance(value, (QtCore.QDate, QtCore.QDateTime, QtCore.QTime)):
        return value.toString(QtCore.Qt.ISODate)
    elif isinstance(value, QtCore.QPoint):
        return '(%d,%d)' % (value.x(), value.y())
    elif isinstance(value, QtCore.QRect):
        return '(%d,%d,%d,%d)' % (value.x(), value.y(), value.width(), value.height())
    elif isinstance(value, QtCore.QSize):
        return '(%d,%d)' % (value.width(), value.height())
    elif isinstance(value, list):
        return ','.join(value)
    elif value is None:
        return '<Invalid>'
    return '<%s>' % value