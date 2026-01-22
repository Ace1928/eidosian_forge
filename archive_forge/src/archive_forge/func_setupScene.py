import math
from PySide2 import QtCore, QtGui, QtWidgets
def setupScene(self):
    self.m_scene.setSceneRect(-300, -200, 600, 460)
    linearGrad = QtGui.QLinearGradient(QtCore.QPointF(-100, -100), QtCore.QPointF(100, 100))
    linearGrad.setColorAt(0, QtGui.QColor(255, 255, 255))
    linearGrad.setColorAt(1, QtGui.QColor(192, 192, 255))
    self.setBackgroundBrush(linearGrad)
    radialGrad = QtGui.QRadialGradient(30, 30, 30)
    radialGrad.setColorAt(0, QtCore.Qt.yellow)
    radialGrad.setColorAt(0.2, QtCore.Qt.yellow)
    radialGrad.setColorAt(1, QtCore.Qt.transparent)
    pixmap = QtGui.QPixmap(60, 60)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(radialGrad)
    painter.drawEllipse(0, 0, 60, 60)
    painter.end()
    self.m_lightSource = self.m_scene.addPixmap(pixmap)
    self.m_lightSource.setZValue(2)
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i + j & 1:
                item = QtWidgets.QGraphicsEllipseItem(0, 0, 50, 50)
            else:
                item = QtWidgets.QGraphicsRectItem(0, 0, 50, 50)
            item.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            item.setBrush(QtGui.QBrush(QtCore.Qt.white))
            effect = QtWidgets.QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(8)
            item.setGraphicsEffect(effect)
            item.setZValue(1)
            item.setPos(i * 80, j * 80)
            self.m_scene.addItem(item)
            self.m_items.append(item)