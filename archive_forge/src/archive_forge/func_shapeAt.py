import random
from PySide2 import QtCore, QtGui, QtWidgets
def shapeAt(self, x, y):
    return self.board[y * TetrixBoard.BoardWidth + x]