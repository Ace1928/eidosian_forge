import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def sceneScaleChanged(self, scale):
    newScale = int(scale[:-1]) / 100.0
    oldMatrix = self.view.matrix()
    self.view.resetMatrix()
    self.view.translate(oldMatrix.dx(), oldMatrix.dy())
    self.view.scale(newScale, newScale)