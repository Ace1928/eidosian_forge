import random
from PySide2 import QtCore, QtGui, QtWidgets
def dropDown(self):
    dropHeight = 0
    newY = self.curY
    while newY > 0:
        if not self.tryMove(self.curPiece, self.curX, newY - 1):
            break
        newY -= 1
        dropHeight += 1
    self.pieceDropped(dropHeight)