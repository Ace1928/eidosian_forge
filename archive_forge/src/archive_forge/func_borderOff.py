from ..Qt import QtCore, QtWidgets
def borderOff(self):
    self.setStyleSheet()
    self.count += 1
    if self.count >= 2:
        if self.limitedTime:
            return
    QtCore.QTimer.singleShot(30, self.borderOn)