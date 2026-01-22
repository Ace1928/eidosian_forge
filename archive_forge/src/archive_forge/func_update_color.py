from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def update_color(self):
    color = self.text()
    qcolor = to_qcolor(color)
    self.colorbtn.color = qcolor