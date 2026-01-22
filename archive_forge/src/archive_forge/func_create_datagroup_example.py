from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def create_datagroup_example():
    datalist = create_datalist_example()
    return ((datalist, 'Category 1', 'Category 1 comment'), (datalist, 'Category 2', 'Category 2 comment'), (datalist, 'Category 3', 'Category 3 comment'))