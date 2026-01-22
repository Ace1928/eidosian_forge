import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
def updateRange(self, vb, viewRange):
    origin = self.tickSize / 2.0
    length = self.length
    lengthIncludingPadding = length + self.tickSize + 2
    self._range = viewRange
    for pos in self.all_ticks:
        tickValueIncludingPadding = (pos - viewRange[0]) / (viewRange[1] - viewRange[0])
        tickValue = (tickValueIncludingPadding * lengthIncludingPadding - origin) / length
        visible = bool(tickValue >= 0 and tickValue <= 1)
        tick = self.all_ticks[pos]
        tick.setVisible(visible)
        if visible:
            self.setTickValue(tick, tickValue)