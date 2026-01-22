from ..Qt import QtCore, QtWidgets
def setMultiline(self, ml):
    if ml:
        self.setPlaceholderText(self.ps2)
    else:
        self.setPlaceholderText(self.ps1)