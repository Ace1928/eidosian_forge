import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def clearChildren(self):
    for c in self.children:
        c.clearChildren()
        c.setParentItem(None)
        self.scene().removeItem(c)
    self.children = []