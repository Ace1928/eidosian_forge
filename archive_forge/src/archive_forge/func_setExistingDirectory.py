import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setExistingDirectory(self):
    options = QtWidgets.QFileDialog.DontResolveSymlinks | QtWidgets.QFileDialog.ShowDirsOnly
    directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'QFileDialog.getExistingDirectory()', self.directoryLabel.text(), options)
    if directory:
        self.directoryLabel.setText(directory)