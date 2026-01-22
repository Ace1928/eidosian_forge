from ..Qt import QtCore, QtGui, QtWidgets
def requestRemove(self):
    QtCore.QTimer.singleShot(0, self.param.remove)