from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
def movePlayer(self, direction):
    if direction == self.Left:
        if self.map[self.pX - 1][self.pY] != '#':
            self.pX -= 1
    elif direction == self.Right:
        if self.map[self.pX + 1][self.pY] != '#':
            self.pX += 1
    elif direction == self.Up:
        if self.map[self.pX][self.pY - 1] != '#':
            self.pY -= 1
    elif direction == self.Down:
        if self.map[self.pX][self.pY + 1] != '#':
            self.pY += 1
    self.repaint()