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
def setAnimation(self, a):
    if a:
        self.lastAnimTime = perf_counter()
        self.animTimer.start(int(self.animDt * 1000))
    else:
        self.animTimer.stop()