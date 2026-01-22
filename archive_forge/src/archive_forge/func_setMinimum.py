from time import perf_counter
from ..Qt import QtCore, QtGui, QtWidgets
def setMinimum(self, val):
    if self.disabled:
        return
    QtWidgets.QProgressDialog.setMinimum(self, val)