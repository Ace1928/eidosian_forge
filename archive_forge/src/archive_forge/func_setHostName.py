from ..Qt import QtCore, QtWidgets
import weakref
def setHostName(self, name):
    self.hostName = name
    self.updateCanvasList()