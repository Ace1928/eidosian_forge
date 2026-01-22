from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def qfont_to_tuple(font):
    return (str(font.family()), int(font.pointSize()), font.italic(), font.bold())