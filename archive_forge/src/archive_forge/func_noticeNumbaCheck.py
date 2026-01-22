import argparse
import itertools
import sys
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import VideoTemplate_generic as ui_template
def noticeNumbaCheck():
    pg.setConfigOption('useNumba', _has_numba and ui.numbaCheck.isChecked())