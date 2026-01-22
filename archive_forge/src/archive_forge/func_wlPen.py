import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def wlPen(wl):
    """Return a pen representing the given wavelength"""
    l1 = 400
    l2 = 700
    hue = np.clip((l2 - l1 - (wl - l1)) * 0.8 / (l2 - l1), 0, 0.8)
    val = 1.0
    if wl > 700:
        val = 1.0 * ((700 - wl) / 700.0 + 1)
    elif wl < 400:
        val = wl * 1.0 / 400.0
    color = pg.hsvColor(hue, 1.0, val)
    pen = pg.mkPen(color)
    return pen