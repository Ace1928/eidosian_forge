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
def updatePlot(self):
    self.plot.clear()
    if self.data is None or len(self.data) == 0:
        return
    if self.filtered is None:
        mask = self.filter.generateMask(self.data)
        self.filtered = self.data[mask]
        self.filteredIndices = self.indices[mask]
    data = self.filtered
    if len(data) == 0:
        return
    colors = np.array([fn.mkBrush(*x) for x in self.colorMap.map(data)])
    style = self.style.copy()
    sel = list([str(item.text()) for item in self.fieldList.selectedItems()])
    units = list([item.opts.get('units', '') for item in self.fieldList.selectedItems()])
    if len(sel) == 0:
        self.plot.setTitle('')
        return
    if len(sel) == 1:
        self.plot.setLabels(left=('N', ''), bottom=(sel[0], units[0]), title='')
        if len(data) == 0:
            return
        xy = [data[sel[0]], None]
    elif len(sel) == 2:
        self.plot.setLabels(left=(sel[1], units[1]), bottom=(sel[0], units[0]))
        if len(data) == 0:
            return
        xy = [data[sel[0]], data[sel[1]]]
    enum = [False, False]
    for i in [0, 1]:
        axis = self.plot.getAxis(['bottom', 'left'][i])
        if xy[i] is not None and (self.fields[sel[i]].get('mode', None) == 'enum' or xy[i].dtype.kind in ('S', 'O')):
            vals = self.fields[sel[i]].get('values', list(set(xy[i])))
            xy[i] = np.array([vals.index(x) if x in vals else len(vals) for x in xy[i]], dtype=float)
            axis.setTicks([list(enumerate(vals))])
            enum[i] = True
        else:
            axis.setTicks(None)
    mask = np.ones(len(xy[0]), dtype=bool)
    if xy[0].dtype.kind == 'f':
        mask &= np.isfinite(xy[0])
    if xy[1] is not None and xy[1].dtype.kind == 'f':
        mask &= np.isfinite(xy[1])
    xy[0] = xy[0][mask]
    style['symbolBrush'] = colors[mask]
    data = data[mask]
    indices = self.filteredIndices[mask]
    if xy[1] is None:
        xy[1] = fn.pseudoScatter(xy[0])
    else:
        xy[1] = xy[1][mask]
        for ax in [0, 1]:
            if not enum[ax]:
                continue
            imax = int(xy[ax].max()) if len(xy[ax]) > 0 else 0
            for i in range(imax + 1):
                keymask = xy[ax] == i
                scatter = fn.pseudoScatter(xy[1 - ax][keymask], bidir=True)
                if len(scatter) == 0:
                    continue
                smax = np.abs(scatter).max()
                if smax != 0:
                    scatter *= 0.2 / smax
                xy[ax][keymask] += scatter
    if self.scatterPlot is not None:
        try:
            self.scatterPlot.sigPointsClicked.disconnect(self.plotClicked)
        except:
            pass
    self._visibleXY = xy
    self._visibleData = data
    self._visibleIndices = indices
    self._indexMap = None
    self.scatterPlot = self.plot.plot(xy[0], xy[1], data=data, **style)
    self.scatterPlot.sigPointsClicked.connect(self.plotClicked)
    self.scatterPlot.sigPointsHovered.connect(self.plotHovered)
    self.updateSelected()