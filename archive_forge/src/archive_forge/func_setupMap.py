from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
def setupMap(self):
    self.map = []
    qsrand(QTime(0, 0, 0).secsTo(QTime.currentTime()))
    for x in range(self.width):
        column = []
        for y in range(self.height):
            if x == 0 or x == self.width - 1 or y == 0 or (y == self.height - 1) or (qrand() % 40 == 0):
                column.append('#')
            else:
                column.append('.')
        self.map.append(column)