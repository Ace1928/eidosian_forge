import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
def updateUI(self):

    def set_colors(window_color, text_color):
        pal = self.window().palette()
        pal.setColor(QPalette.Window, window_color)
        pal.setColor(QPalette.WindowText, text_color)
        self.window().setPalette(pal)
    idx = self.ui.themeComboBox.currentIndex()
    theme = self.ui.themeComboBox.itemData(idx)
    if len(self.charts):
        chart_theme = self.charts[0].chart().theme()
        if chart_theme != theme:
            for chart_view in self.charts:
                if theme == 0:
                    theme_name = QtCharts.QChart.ChartThemeLight
                elif theme == 1:
                    theme_name = QtCharts.QChart.ChartThemeBlueCerulean
                elif theme == 2:
                    theme_name = QtCharts.QChart.ChartThemeDark
                elif theme == 3:
                    theme_name = QtCharts.QChart.ChartThemeBrownSand
                elif theme == 4:
                    theme_name = QtCharts.QChart.ChartThemeBlueNcs
                elif theme == 5:
                    theme_name = QtCharts.QChart.ChartThemeHighContrast
                elif theme == 6:
                    theme_name = QtCharts.QChart.ChartThemeBlueIcy
                elif theme == 7:
                    theme_name = QtCharts.QChart.ChartThemeQt
                else:
                    theme_name = QtCharts.QChart.ChartThemeLight
                chart_view.chart().setTheme(theme_name)
            if theme == QtCharts.QChart.ChartThemeLight:
                set_colors(QColor(15790320), QColor(4210756))
            elif theme == QtCharts.QChart.ChartThemeDark:
                set_colors(QColor(1184280), QColor(14079702))
            elif theme == QtCharts.QChart.ChartThemeBlueCerulean:
                set_colors(QColor(4211530), QColor(14079702))
            elif theme == QtCharts.QChart.ChartThemeBrownSand:
                set_colors(QColor(10389861), QColor(4210756))
            elif theme == QtCharts.QChart.ChartThemeBlueNcs:
                set_colors(QColor(101306), QColor(4210756))
            elif theme == QtCharts.QChart.ChartThemeHighContrast:
                set_colors(QColor(16755459), QColor(1579032))
            elif theme == QtCharts.QChart.ChartThemeBlueIcy:
                set_colors(QColor(13559792), QColor(4210756))
            else:
                set_colors(QColor(15790320), QColor(4210756))
    checked = self.ui.antialiasCheckBox.isChecked()
    for chart in self.charts:
        chart.setRenderHint(QPainter.Antialiasing, checked)
    idx = self.ui.animatedComboBox.currentIndex()
    options = self.ui.animatedComboBox.itemData(idx)
    if len(self.charts):
        chart = self.charts[0].chart()
        animation_options = chart.animationOptions()
        if animation_options != options:
            for chart_view in self.charts:
                options_name = QtCharts.QChart.NoAnimation
                if options == 0:
                    options_name = QtCharts.QChart.NoAnimation
                elif options == 1:
                    options_name = QtCharts.QChart.GridAxisAnimations
                elif options == 2:
                    options_name = QtCharts.QChart.SeriesAnimations
                elif options == 3:
                    options_name = QtCharts.QChart.AllAnimations
                chart_view.chart().setAnimationOptions(options_name)
    idx = self.ui.legendComboBox.currentIndex()
    alignment = self.ui.legendComboBox.itemData(idx)
    if not alignment:
        for chart_view in self.charts:
            chart_view.chart().legend().hide()
    else:
        for chart_view in self.charts:
            alignment_name = Qt.AlignTop
            if alignment == 32:
                alignment_name = Qt.AlignTop
            elif alignment == 64:
                alignment_name = Qt.AlignBottom
            elif alignment == 1:
                alignment_name = Qt.AlignLeft
            elif alignment == 2:
                alignment_name = Qt.AlignRight
            chart_view.chart().legend().setAlignment(alignment_name)
            chart_view.chart().legend().show()