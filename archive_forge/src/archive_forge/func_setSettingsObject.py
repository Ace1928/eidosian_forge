import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setSettingsObject(self, settings):
    self.settings = settings
    self.clear()
    if self.settings is not None:
        self.settings.setParent(self)
        self.refresh()
        if self.autoRefresh:
            self.refreshTimer.start()
    else:
        self.refreshTimer.stop()