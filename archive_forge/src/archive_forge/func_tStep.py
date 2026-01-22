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
@staticmethod
def tStep(dt, v0, x0, tau0, g):
    gamma = (1.0 - v0 ** 2) ** (-0.5)
    dtau = dt / gamma
    return (v0 + dtau * g, x0 + v0 * dt, tau0 + dtau)