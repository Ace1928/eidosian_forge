import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def setGreen(self):
    self.pen = pg.mkPen('g')
    self.update()