import pickle
from PySide2 import QtCore, QtGui, QtWidgets
def findContact(self):
    self.dialog.show()
    if self.dialog.exec_() == QtWidgets.QDialog.Accepted:
        contactName = self.dialog.getFindText()
        if contactName in self.contacts:
            self.nameLine.setText(contactName)
            self.addressText.setText(self.contacts[contactName])
        else:
            QtWidgets.QMessageBox.information(self, 'Contact Not Found', 'Sorry, "%s" is not in your address book.' % contactName)
            return
    self.updateInterface(self.NavigationMode)