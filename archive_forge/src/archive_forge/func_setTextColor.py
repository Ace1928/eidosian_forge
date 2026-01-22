import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def setTextColor(self, color):
    self.myTextColor = color
    if self.isItemChange(DiagramTextItem):
        item = self.selectedItems()[0]
        item.setDefaultTextColor(self.myTextColor)