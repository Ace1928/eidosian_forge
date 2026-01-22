import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def updateTransform(self):
    self.setPos(0, 0)
    tr = QtGui.QTransform()
    self.setTransform(tr.translate(Point(self['pos'])).rotate(self['angle']))