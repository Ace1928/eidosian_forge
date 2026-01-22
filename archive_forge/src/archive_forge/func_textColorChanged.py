import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def textColorChanged(self):
    self.textAction = self.sender()
    self.fontColorToolButton.setIcon(self.createColorToolButtonIcon(':/images/textpointer.png', QtGui.QColor(self.textAction.data())))
    self.textButtonTriggered()