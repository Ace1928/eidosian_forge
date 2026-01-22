import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def fillButtonTriggered(self):
    self.scene.setItemColor(QtGui.QColor(self.fillAction.data()))