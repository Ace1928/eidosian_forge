from PySide2 import QtCore, QtGui, QtWidgets
import application_rc
def documentWasModified(self):
    self.setWindowModified(self.textEdit.document().isModified())