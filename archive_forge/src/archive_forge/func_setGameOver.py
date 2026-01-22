import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
def setGameOver(self):
    if self.gameEnded:
        return
    if self.isShooting():
        self.autoShootTimer.stop()
    self.gameEnded = True
    self.update()