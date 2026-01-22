import random
from PySide2 import QtCore, QtGui, QtWidgets
def removeFullLines(self):
    numFullLines = 0
    for i in range(TetrixBoard.BoardHeight - 1, -1, -1):
        lineIsFull = True
        for j in range(TetrixBoard.BoardWidth):
            if self.shapeAt(j, i) == NoShape:
                lineIsFull = False
                break
        if lineIsFull:
            numFullLines += 1
            for k in range(TetrixBoard.BoardHeight - 1):
                for j in range(TetrixBoard.BoardWidth):
                    self.setShapeAt(j, k, self.shapeAt(j, k + 1))
            for j in range(TetrixBoard.BoardWidth):
                self.setShapeAt(j, TetrixBoard.BoardHeight - 1, NoShape)
    if numFullLines > 0:
        self.numLinesRemoved += numFullLines
        self.score += 10 * numFullLines
        self.linesRemovedChanged.emit(self.numLinesRemoved)
        self.scoreChanged.emit(self.score)
        self.timer.start(500, self)
        self.isWaitingAfterLine = True
        self.curPiece.setShape(NoShape)
        self.update()