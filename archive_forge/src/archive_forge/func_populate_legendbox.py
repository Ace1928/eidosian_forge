import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def populate_legendbox(self):
    legend = self.ui.legendComboBox
    legend.addItem('No Legend ', 0)
    legend.addItem('Legend Top', Qt.AlignTop)
    legend.addItem('Legend Bottom', Qt.AlignBottom)
    legend.addItem('Legend Left', Qt.AlignLeft)
    legend.addItem('Legend Right', Qt.AlignRight)