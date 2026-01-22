import argparse
import itertools
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
import pyqtgraph.functions as fn
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def setMethod(self, value):
    self.monkey_mode = value