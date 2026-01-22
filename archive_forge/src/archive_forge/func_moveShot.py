import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
@QtCore.Slot()
def moveShot(self):
    region = QtGui.QRegion(self.shotRect())
    self.timerCount += 1
    shotR = self.shotRect()
    if shotR.x() > self.width() or shotR.y() > self.height():
        self.autoShootTimer.stop()
    else:
        region = region.united(QtGui.QRegion(shotR))
    self.update(region)