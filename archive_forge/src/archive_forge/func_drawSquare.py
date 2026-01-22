import random
from PySide2 import QtCore, QtGui, QtWidgets
def drawSquare(self, painter, x, y, shape):
    colorTable = [0, 13395558, 6736998, 6710988, 13421670, 13395660, 6737100, 14330368]
    color = QtGui.QColor(colorTable[shape])
    painter.fillRect(x + 1, y + 1, self.squareWidth() - 2, self.squareHeight() - 2, color)
    painter.setPen(color.lighter())
    painter.drawLine(x, y + self.squareHeight() - 1, x, y)
    painter.drawLine(x, y, x + self.squareWidth() - 1, y)
    painter.setPen(color.darker())
    painter.drawLine(x + 1, y + self.squareHeight() - 1, x + self.squareWidth() - 1, y + self.squareHeight() - 1)
    painter.drawLine(x + self.squareWidth() - 1, y + self.squareHeight() - 1, x + self.squareWidth() - 1, y + 1)