from PySide2 import QtCore, QtGui, QtWidgets
def updateInterface(self, mode):
    self.currentMode = mode
    if self.currentMode in (self.AddingMode, self.EditingMode):
        self.nameLine.setReadOnly(False)
        self.nameLine.setFocus(QtCore.Qt.OtherFocusReason)
        self.addressText.setReadOnly(False)
        self.addButton.setEnabled(False)
        self.editButton.setEnabled(False)
        self.removeButton.setEnabled(False)
        self.nextButton.setEnabled(False)
        self.previousButton.setEnabled(False)
        self.submitButton.show()
        self.cancelButton.show()
    elif self.currentMode == self.NavigationMode:
        if not self.contacts:
            self.nameLine.clear()
            self.addressText.clear()
        self.nameLine.setReadOnly(True)
        self.addressText.setReadOnly(True)
        self.addButton.setEnabled(True)
        number = len(self.contacts)
        self.editButton.setEnabled(number >= 1)
        self.removeButton.setEnabled(number >= 1)
        self.nextButton.setEnabled(number > 1)
        self.previousButton.setEnabled(number > 1)
        self.submitButton.hide()
        self.cancelButton.hide()