import sys
from PySide2 import QtCore, QtGui, QtWidgets
def childCount(self, parent):
    if parent is not None:
        return parent.childCount()
    else:
        return self.topLevelItemCount()