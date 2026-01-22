import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def create_scatterchart(self):
    chart = QtCharts.QChart()
    chart.setTitle('Scatter chart')
    name = 'Series '
    for i, lst in enumerate(self.data_table):
        series = QtCharts.QScatterSeries(chart)
        for data in lst:
            series.append(data[0])
        series.setName('{}{}'.format(name, i))
        chart.addSeries(series)
    chart.createDefaultAxes()
    chart.axisX().setRange(0, self.value_max)
    chart.axisY().setRange(0, self.value_count)
    chart.axisY().setLabelFormat('%.1f  ')
    return chart