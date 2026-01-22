import random
from PySide2 import QtCore, QtGui, QtWidgets
def newPiece(self):
    self.curPiece = self.nextPiece
    self.nextPiece.setRandomShape()
    self.showNextPiece()
    self.curX = TetrixBoard.BoardWidth // 2 + 1
    self.curY = TetrixBoard.BoardHeight - 1 + self.curPiece.minY()
    if not self.tryMove(self.curPiece, self.curX, self.curY):
        self.curPiece.setShape(NoShape)
        self.timer.stop()
        self.isStarted = False