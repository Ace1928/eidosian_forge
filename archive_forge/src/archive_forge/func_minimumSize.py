from PySide2 import QtCore, QtGui, QtWidgets
def minimumSize(self):
    size = QtCore.QSize()
    for item in self.itemList:
        size = size.expandedTo(item.minimumSize())
    size += QtCore.QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
    return size