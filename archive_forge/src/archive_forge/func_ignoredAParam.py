from functools import wraps
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import (
@interactor.decorate(ignores=['a'])
@printResult
def ignoredAParam(a=10, b=20):
    return a * b