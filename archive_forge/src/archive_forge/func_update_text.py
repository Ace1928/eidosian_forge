from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def update_text(self, color):
    self.lineedit.setText(mcolors.to_hex(color.getRgbF(), keep_alpha=True))