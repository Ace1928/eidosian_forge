import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
def hoverLeaveEvent(self, ev):
    self.setPen(self.savedPen)
    ev.ignore()