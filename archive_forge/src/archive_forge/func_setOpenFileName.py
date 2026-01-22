import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setOpenFileName(self):
    options = QtWidgets.QFileDialog.Options()
    if not self.native.isChecked():
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
    fileName, filtr = QtWidgets.QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', self.openFileNameLabel.text(), 'All Files (*);;Text Files (*.txt)', '', options)
    if fileName:
        self.openFileNameLabel.setText(fileName)