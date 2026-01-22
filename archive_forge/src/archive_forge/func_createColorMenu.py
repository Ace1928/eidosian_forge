import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def createColorMenu(self, slot, defaultColor):
    colors = [QtCore.Qt.black, QtCore.Qt.white, QtCore.Qt.red, QtCore.Qt.blue, QtCore.Qt.yellow]
    names = ['black', 'white', 'red', 'blue', 'yellow']
    colorMenu = QtWidgets.QMenu(self)
    for color, name in zip(colors, names):
        action = QtWidgets.QAction(self.createColorIcon(color), name, self, triggered=slot)
        action.setData(QtGui.QColor(color))
        colorMenu.addAction(action)
        if color == defaultColor:
            colorMenu.setDefaultAction(action)
    return colorMenu