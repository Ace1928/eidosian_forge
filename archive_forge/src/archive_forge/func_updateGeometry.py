import sys
from PySide2.QtWidgets import (QApplication, QWidget, QGraphicsScene,
from PySide2.QtCore import Qt, QPointF, QRectF, QRect
from PySide2.QtCharts import QtCharts
from PySide2.QtGui import QPainter, QFont, QFontMetrics, QPainterPath, QColor
def updateGeometry(self):
    self.prepareGeometryChange()
    self.setPos(self._chart.mapToPosition(self._anchor) + QPointF(10, -50))