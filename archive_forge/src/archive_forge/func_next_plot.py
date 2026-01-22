import traceback
import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import name_list
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import interact, ParameterTree, Parameter
import random
def next_plot(xtype='random', ytype='random', symbol='o', symbolBrush='#f00'):
    constKwargs = locals()
    x = y = None
    if xtype == 'random':
        xtype = random.choice(list(values))
    if ytype == 'random':
        ytype = random.choice(list(values))
    x = values[xtype]
    y = values[ytype]
    textbox.setValue(f'x={xtype}\ny={ytype}')
    pltItem.clear()
    try:
        pltItem.multiDataPlot(x=x, y=y, pen=cmap.getLookupTable(nPts=6), constKwargs=constKwargs)
    except Exception as e:
        QtWidgets.QMessageBox.critical(widget, 'Error', traceback.format_exc())