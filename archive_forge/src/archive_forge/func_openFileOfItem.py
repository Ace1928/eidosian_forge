from PySide2 import QtCore, QtGui, QtWidgets
def openFileOfItem(self, row, column):
    item = self.filesTable.item(row, 0)
    QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.currentDir.absoluteFilePath(item.text())))