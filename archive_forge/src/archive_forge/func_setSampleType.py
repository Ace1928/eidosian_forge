import math
from .. import functions as fn
from ..icons import invisibleEye
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .BarGraphItem import BarGraphItem
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .LabelItem import LabelItem
from .PlotDataItem import PlotDataItem
from .ScatterPlotItem import ScatterPlotItem, drawSymbol
def setSampleType(self, sample):
    """Set the new sample item claspes"""
    if sample is self.sampleType:
        return
    items = list(self.items)
    self.sampleType = sample
    self.clear()
    for sample, label in items:
        plot_item = sample.item
        plot_name = label.text
        self.addItem(plot_item, plot_name)
    self.updateSize()