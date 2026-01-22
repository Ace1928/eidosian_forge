import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def createPieChart(self):
    chart = QtCharts.QChart()
    chart.setTitle('Pie chart')
    series = QtCharts.QPieSeries(chart)
    for data in self.data_table[0]:
        slc = series.append(data[1], data[0].y())
        if data == self.data_table[0][0]:
            slc.setLabelVisible()
            slc.setExploded()
            slc.setExplodeDistanceFactor(0.5)
    series.setPieSize(0.4)
    chart.addSeries(series)
    return chart