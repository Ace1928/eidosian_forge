import pickle
from PySide2 import QtCore, QtGui, QtWidgets
def findClicked(self):
    text = self.lineEdit.text()
    if not text:
        QtWidgets.QMessageBox.information(self, 'Empty Field', 'Please enter a name.')
        return
    self.findText = text
    self.lineEdit.clear()
    self.hide()