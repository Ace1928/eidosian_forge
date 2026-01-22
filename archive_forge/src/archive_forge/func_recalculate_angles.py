import sys
from PySide2.QtCore import Qt
from PySide2.QtGui import QColor, QFont, QPainter
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCharts import QtCharts
def recalculate_angles(self):
    angle = 0
    slices = self.main_series.slices()
    for pie_slice in slices:
        breakdown_series = pie_slice.get_breakdown_series()
        breakdown_series.setPieStartAngle(angle)
        angle += pie_slice.percentage() * 360.0
        breakdown_series.setPieEndAngle(angle)