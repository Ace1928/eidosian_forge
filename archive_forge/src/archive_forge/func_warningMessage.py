import sys
from PySide2 import QtCore, QtGui, QtWidgets
def warningMessage(self):
    msgBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, 'QMessageBox.warning()', Dialog.MESSAGE, QtWidgets.QMessageBox.NoButton, self)
    msgBox.addButton('Save &Again', QtWidgets.QMessageBox.AcceptRole)
    msgBox.addButton('&Continue', QtWidgets.QMessageBox.RejectRole)
    if msgBox.exec_() == QtWidgets.QMessageBox.AcceptRole:
        self.warningLabel.setText('Save Again')
    else:
        self.warningLabel.setText('Continue')