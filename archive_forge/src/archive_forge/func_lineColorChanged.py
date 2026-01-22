import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def lineColorChanged(self):
    self.lineAction = self.sender()
    self.lineColorToolButton.setIcon(self.createColorToolButtonIcon(':/images/linecolor.png', QtGui.QColor(self.lineAction.data())))
    self.lineButtonTriggered()