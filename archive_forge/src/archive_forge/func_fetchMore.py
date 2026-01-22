from PySide2 import QtCore, QtGui, QtWidgets
def fetchMore(self, index):
    remainder = len(self.fileList) - self.fileCount
    itemsToFetch = min(100, remainder)
    self.beginInsertRows(QtCore.QModelIndex(), self.fileCount, self.fileCount + itemsToFetch)
    self.fileCount += itemsToFetch
    self.endInsertRows()
    self.numberPopulated.emit(itemsToFetch)