import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def runInStack3(func):
    x = 'inside runInStack3(func)'
    runInStack4(func)
    return x