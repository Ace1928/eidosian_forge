import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def update_legend_layout(self):
    legend = self.chart.legend()
    rect = QRectF(self.legend_posx.value(), self.legend_posy.value(), self.legend_width.value(), self.legend_height.value())
    legend.setGeometry(rect)
    legend.update()