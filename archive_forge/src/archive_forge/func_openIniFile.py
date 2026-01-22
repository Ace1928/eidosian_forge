import sys
from PySide2 import QtCore, QtGui, QtWidgets
def openIniFile(self):
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open INI File', '', 'INI Files (*.ini *.conf)')
    if fileName:
        settings = QtCore.QSettings(fileName, QtCore.QSettings.IniFormat)
        self.setSettingsObject(settings)
        self.fallbacksAct.setEnabled(False)