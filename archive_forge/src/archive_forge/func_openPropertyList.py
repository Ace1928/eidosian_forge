import sys
from PySide2 import QtCore, QtGui, QtWidgets
def openPropertyList(self):
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Property List', '', 'Property List Files (*.plist)')
    if fileName:
        settings = QtCore.QSettings(fileName, QtCore.QSettings.NativeFormat)
        self.setSettingsObject(settings)
        self.fallbacksAct.setEnabled(False)