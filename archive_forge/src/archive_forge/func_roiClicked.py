import os
from math import log10
from time import perf_counter
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..graphicsItems.GradientEditorItem import addGradientListToDocstring
from ..graphicsItems.ImageItem import ImageItem
from ..graphicsItems.InfiniteLine import InfiniteLine
from ..graphicsItems.LinearRegionItem import LinearRegionItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..graphicsItems.VTickGroup import VTickGroup
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
from . import ImageViewTemplate_generic as ui_template
def roiClicked(self):
    showRoiPlot = False
    if self.ui.roiBtn.isChecked():
        showRoiPlot = True
        self.roi.show()
        self.ui.roiPlot.setMouseEnabled(True, True)
        self.ui.splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])
        self.ui.splitter.handle(1).setEnabled(True)
        self.roiChanged()
        for c in self.roiCurves:
            c.show()
        self.ui.roiPlot.showAxis('left')
    else:
        self.roi.hide()
        self.ui.roiPlot.setMouseEnabled(False, False)
        for c in self.roiCurves:
            c.hide()
        self.ui.roiPlot.hideAxis('left')
    if self.hasTimeAxis():
        showRoiPlot = True
        mn = self.tVals.min()
        mx = self.tVals.max()
        self.ui.roiPlot.setXRange(mn, mx, padding=0.01)
        self.timeLine.show()
        self.timeLine.setBounds([mn, mx])
        if not self.ui.roiBtn.isChecked():
            self.ui.splitter.setSizes([self.height() - 35, 35])
            self.ui.splitter.handle(1).setEnabled(False)
    else:
        self.timeLine.hide()
    self.ui.roiPlot.setVisible(showRoiPlot)