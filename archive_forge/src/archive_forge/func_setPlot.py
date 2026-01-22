import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode
def setPlot(self, plot):
    if plot == self.plot:
        return
    if self.plot is not None:
        for vid in list(self.items.keys()):
            self.plot.removeItem(self.items[vid])
            del self.items[vid]
    self.plot = plot
    self.updateUi()
    self.update()
    self.sigPlotChanged.emit(self)