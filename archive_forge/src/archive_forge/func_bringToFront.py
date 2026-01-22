import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def bringToFront(self):
    if not self.scene.selectedItems():
        return
    selectedItem = self.scene.selectedItems()[0]
    overlapItems = selectedItem.collidingItems()
    zValue = 0
    for item in overlapItems:
        if item.zValue() >= zValue and isinstance(item, DiagramItem):
            zValue = item.zValue() + 0.1
    selectedItem.setZValue(zValue)