import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode
def updateUi(self):
    self.ui.setItems(self.plots)
    try:
        self.ui.setValue(self.plot)
    except ValueError:
        pass