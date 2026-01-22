import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def removeArrows(self):
    for arrow in self.arrows[:]:
        arrow.startItem().removeArrow(arrow)
        arrow.endItem().removeArrow(arrow)
        self.scene().removeItem(arrow)