import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def paintCannon(self, painter):
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtCore.Qt.blue)
    painter.save()
    painter.translate(0, self.height())
    painter.drawPie(QtCore.QRect(-35, -35, 70, 70), 0, 90 * 16)
    painter.rotate(-self.currentAngle)
    painter.drawRect(CannonField.barrelRect)
    painter.restore()