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
def loadState(self, state):
    if 'Load Preset..' in state['children']:
        del state['children']['Load Preset..']['limits']
        del state['children']['Load Preset..']['value']
    self.params.param('Objects').clearChildren()
    self.params.restoreState(state, removeChildren=False)
    self.recalculate()