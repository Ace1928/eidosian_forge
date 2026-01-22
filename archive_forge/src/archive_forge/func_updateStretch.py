import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
def updateStretch(self):
    x = 0
    y = 0
    for i in range(self.count()):
        wx, wy = self.widget(i).stretch()
        x = max(x, wx)
        y = max(y, wy)
    self.setStretch(x, y)