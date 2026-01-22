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
def hypIntersect(x0r, t0r, vr, x0, t0, v0, g):
    if g == 0:
        t = (-t0r + t0 * v0 * vr - vr * x0 + vr * x0r) / (-1 + v0 * vr)
        return t
    gamma = (1.0 - v0 ** 2) ** (-0.5)
    sel = (1 if g > 0 else 0) + (1 if vr < 0 else 0)
    sel = sel % 2
    if sel == 0:
        t = 1.0 / (g ** 2 * (-1.0 + vr ** 2)) * (-g ** 2 * t0r + g * gamma * vr + g ** 2 * t0 * vr ** 2 - g * gamma * v0 * vr ** 2 - g ** 2 * vr * x0 + g ** 2 * vr * x0r + np.sqrt(g ** 2 * vr ** 2 * (1.0 + gamma ** 2 * (v0 - vr) ** 2 - vr ** 2 + 2 * g * gamma * (v0 - vr) * (-t0 + t0r + vr * (x0 - x0r)) + g ** 2 * (t0 - t0r + vr * (-x0 + x0r)) ** 2)))
    else:
        t = -(1.0 / (g ** 2 * (-1.0 + vr ** 2))) * (g ** 2 * t0r - g * gamma * vr - g ** 2 * t0 * vr ** 2 + g * gamma * v0 * vr ** 2 + g ** 2 * vr * x0 - g ** 2 * vr * x0r + np.sqrt(g ** 2 * vr ** 2 * (1.0 + gamma ** 2 * (v0 - vr) ** 2 - vr ** 2 + 2 * g * gamma * (v0 - vr) * (-t0 + t0r + vr * (x0 - x0r)) + g ** 2 * (t0 - t0r + vr * (-x0 + x0r)) ** 2)))
    return t