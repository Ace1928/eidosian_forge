import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def deleteItem(self):
    for item in self.scene.selectedItems():
        if isinstance(item, DiagramItem):
            item.removeArrows()
        self.scene.removeItem(item)