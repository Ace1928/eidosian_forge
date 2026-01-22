from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import getConfigOption
from .. import parametertree as ptree
from ..graphicsItems.TextItem import TextItem
from ..Qt import QtCore, QtWidgets
from .ColorMapWidget import ColorMapParameter
from .DataFilterWidget import DataFilterParameter
from .PlotWidget import PlotWidget
def updateSelected(self):
    if self._visibleXY is None:
        return
    indMap = self._getIndexMap()
    inds = [indMap[i] for i in self.selectedIndices if i in indMap]
    x, y = (self._visibleXY[0][inds], self._visibleXY[1][inds])
    if self.selectionScatter is not None:
        self.plot.plotItem.removeItem(self.selectionScatter)
    if len(x) == 0:
        return
    self.selectionScatter = self.plot.plot(x, y, pen=None, symbol='s', symbolSize=12, symbolBrush=None, symbolPen='y')