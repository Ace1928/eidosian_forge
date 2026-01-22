import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def setLineColor(self, color):
    self.myLineColor = color
    if self.isItemChange(Arrow):
        item = self.selectedItems()[0]
        item.setColor(self.myLineColor)
        self.update()