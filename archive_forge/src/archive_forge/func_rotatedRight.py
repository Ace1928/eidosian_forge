import random
from PySide2 import QtCore, QtGui, QtWidgets
def rotatedRight(self):
    if self.pieceShape == SquareShape:
        return self
    result = TetrixPiece()
    result.pieceShape = self.pieceShape
    for i in range(4):
        result.setX(i, -self.y(i))
        result.setY(i, self.x(i))
    return result