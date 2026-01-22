import collections
import os
import sys
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def getCurve(self, ref=True):
    if ref is False:
        data = self.inertData
    else:
        data = self.refData[1:]
    x = data['x']
    y = data['t']
    curve = pg.PlotCurveItem(x=x, y=y, pen=self.pen)
    step = 1.0
    inds = [0]
    pt = data['pt']
    for i in range(1, len(pt)):
        diff = pt[i] - pt[inds[-1]]
        if abs(diff) >= step:
            inds.append(i)
    inds = np.array(inds)
    pts = []
    for i in inds:
        x = data['x'][i]
        y = data['t'][i]
        if i + 1 < len(data):
            dpt = data['pt'][i + 1] - data['pt'][i]
            dt = data['t'][i + 1] - data['t'][i]
        else:
            dpt = 1
        if dpt > 0:
            c = pg.mkBrush((0, 0, 0))
        else:
            c = pg.mkBrush((200, 200, 200))
        pts.append({'pos': (x, y), 'brush': c})
    points = pg.ScatterPlotItem(pts, pen=self.pen, size=7)
    return (curve, points)