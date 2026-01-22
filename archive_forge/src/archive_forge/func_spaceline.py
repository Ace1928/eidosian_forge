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
def spaceline(self):
    if self._spaceline is None:
        self._spaceline = pg.InfiniteLine()
        self._spaceline.setPen(self.clock.pen)
    return self._spaceline