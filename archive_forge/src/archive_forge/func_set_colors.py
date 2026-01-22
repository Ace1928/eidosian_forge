import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def set_colors(window_color, text_color):
    pal = self.window().palette()
    pal.setColor(QPalette.Window, window_color)
    pal.setColor(QPalette.WindowText, text_color)
    self.window().setPalette(pal)