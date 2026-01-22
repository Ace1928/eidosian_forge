import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def createCellWidget(self, text, diagramType):
    item = DiagramItem(diagramType, self.itemMenu)
    icon = QtGui.QIcon(item.image())
    button = QtWidgets.QToolButton()
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(50, 50))
    button.setCheckable(True)
    self.buttonGroup.addButton(button, diagramType)
    layout = QtWidgets.QGridLayout()
    layout.addWidget(button, 0, 0, QtCore.Qt.AlignHCenter)
    layout.addWidget(QtWidgets.QLabel(text), 1, 0, QtCore.Qt.AlignCenter)
    widget = QtWidgets.QWidget()
    widget.setLayout(layout)
    return widget