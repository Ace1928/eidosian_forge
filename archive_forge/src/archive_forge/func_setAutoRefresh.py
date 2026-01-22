import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setAutoRefresh(self, autoRefresh):
    self.autoRefresh = autoRefresh
    if self.settings is not None:
        if self.autoRefresh:
            self.maybeRefresh()
            self.refreshTimer.start()
        else:
            self.refreshTimer.stop()