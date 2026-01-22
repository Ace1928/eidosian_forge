import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def generate_random_data(self, list_count, value_max, value_count):
    data_table = []
    for i in range(list_count):
        data_list = []
        y_value = 0
        for j in range(value_count):
            constant = value_max / float(value_count)
            y_value += uniform(0, constant)
            x_value = (j + random()) * constant
            value = QPointF(x_value, y_value)
            label = 'Slice {}: {}'.format(i, j)
            data_list.append((value, label))
        data_table.append(data_list)
    return data_table