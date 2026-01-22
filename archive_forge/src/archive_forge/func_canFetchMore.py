from PySide2 import QtCore, QtGui, QtWidgets
def canFetchMore(self, index):
    return self.fileCount < len(self.fileList)