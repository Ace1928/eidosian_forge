import random
from PySide2 import QtCore, QtGui, QtWidgets
def oneLineDown(self):
    if not self.tryMove(self.curPiece, self.curX, self.curY - 1):
        self.pieceDropped(0)