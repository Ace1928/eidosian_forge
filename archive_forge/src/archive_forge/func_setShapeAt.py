import random
from PySide2 import QtCore, QtGui, QtWidgets
def setShapeAt(self, x, y, shape):
    self.board[y * TetrixBoard.BoardWidth + x] = shape