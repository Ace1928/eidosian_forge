import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def transmissionCurve(self, glass):
    data = self.data[glass]
    keys = [int(x[7:]) for x in data.keys() if 'TAUI25' in x]
    keys.sort()
    curve = np.empty((2, len(keys)))
    for i in range(len(keys)):
        curve[0][i] = keys[i]
        key = 'TAUI25/%d' % keys[i]
        val = data[key]
        if val == '':
            val = 0
        else:
            val = float(val)
        curve[1][i] = val
    return curve