import math
import warnings
import bisect
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem
def updateItems(self, styleUpdate=True):
    styleUpdate = True
    curveArgs = {}
    scatterArgs = {}
    if styleUpdate:
        for k, v in [('pen', 'pen'), ('shadowPen', 'shadowPen'), ('fillLevel', 'fillLevel'), ('fillOutline', 'fillOutline'), ('fillBrush', 'brush'), ('antialias', 'antialias'), ('connect', 'connect'), ('stepMode', 'stepMode'), ('skipFiniteCheck', 'skipFiniteCheck')]:
            if k in self.opts:
                curveArgs[v] = self.opts[k]
        for k, v in [('symbolPen', 'pen'), ('symbolBrush', 'brush'), ('symbol', 'symbol'), ('symbolSize', 'size'), ('data', 'data'), ('pxMode', 'pxMode'), ('antialias', 'antialias'), ('useCache', 'useCache')]:
            if k in self.opts:
                scatterArgs[v] = self.opts[k]
    dataset = self._getDisplayDataset()
    if dataset is None:
        self.curve.hide()
        self.scatter.hide()
        return
    x = dataset.x
    y = dataset.y
    if self.opts['pen'] is not None or (self.opts['fillBrush'] is not None and self.opts['fillLevel'] is not None):
        if isinstance(curveArgs['connect'], str) and curveArgs['connect'] == 'auto':
            if dataset.containsNonfinite is False:
                curveArgs['connect'] = 'all'
                curveArgs['skipFiniteCheck'] = True
            else:
                curveArgs['connect'] = 'finite'
                curveArgs['skipFiniteCheck'] = False
        self.curve.setData(x=x, y=y, **curveArgs)
        self.curve.show()
    else:
        self.curve.hide()
    if self.opts['symbol'] is not None:
        if self.opts.get('stepMode', False) in ('center', True):
            x = 0.5 * (x[:-1] + x[1:])
        self.scatter.setData(x=x, y=y, **scatterArgs)
        self.scatter.show()
    else:
        self.scatter.hide()