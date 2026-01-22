from PySide2 import QtCore, QtGui, QtWidgets
def editContact(self):
    self.oldName = self.nameLine.text()
    self.oldAddress = self.addressText.toPlainText()
    self.updateInterface(self.EditingMode)