import sys
from PySide2 import QtCore, QtGui, QtWidgets
def questionMessage(self):
    reply = QtWidgets.QMessageBox.question(self, 'QMessageBox.question()', Dialog.MESSAGE, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
    if reply == QtWidgets.QMessageBox.Yes:
        self.questionLabel.setText('Yes')
    elif reply == QtWidgets.QMessageBox.No:
        self.questionLabel.setText('No')
    else:
        self.questionLabel.setText('Cancel')