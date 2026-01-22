from functools import wraps
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import (
@interactor.decorate(runOptions=(RunOptions.ON_CHANGED, RunOptions.ON_ACTION), a={'type': 'list', 'limits': [5, 10, 20]})
@printResult
def runOnBtnOrChange_listOpts(a=5):
    return a