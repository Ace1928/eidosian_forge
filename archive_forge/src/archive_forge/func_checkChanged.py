from ..Qt import QtCore, QtWidgets
from . import VerticalLabel
def checkChanged(self, state):
    check = QtCore.QObject.sender(self)
    self.sigStateChanged.emit(check.row, check.col, state)