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
def treeChanged(self, *args):
    clocks = []
    for c in self.params.param('Objects'):
        clocks.extend(c.clockNames())
    self.params.param('Reference Frame').setLimits(clocks)
    self.setAnimation(self.params['Animate'])