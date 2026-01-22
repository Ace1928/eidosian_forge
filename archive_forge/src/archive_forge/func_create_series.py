import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def create_series(self):
    self.add_barset()
    self.add_barset()
    self.add_barset()
    self.add_barset()
    self.chart.addSeries(self.series)
    self.chart.setTitle('Legend detach example')
    self.chart.createDefaultAxes()
    self.chart.legend().setVisible(True)
    self.chart.legend().setAlignment(Qt.AlignBottom)
    self.chart_view.setRenderHint(QPainter.Antialiasing)