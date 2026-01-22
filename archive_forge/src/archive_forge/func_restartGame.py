import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
def restartGame(self):
    if self.isShooting():
        self.autoShootTimer.stop()
    self.gameEnded = False
    self.update()
    self.emit(QtCore.SIGNAL('canShoot(bool)'), True)