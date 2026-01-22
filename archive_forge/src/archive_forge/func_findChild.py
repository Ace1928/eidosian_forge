import sys
from PySide2 import QtCore, QtGui, QtWidgets
def findChild(self, parent, text, startIndex):
    for i in range(self.childCount(parent)):
        if self.childAt(parent, i).text(0) == text:
            return i
    return -1