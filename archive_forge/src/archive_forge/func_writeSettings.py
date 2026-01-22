from PySide2 import QtCore, QtGui, QtWidgets
import application_rc
def writeSettings(self):
    settings = QtCore.QSettings('Trolltech', 'Application Example')
    settings.setValue('pos', self.pos())
    settings.setValue('size', self.size())