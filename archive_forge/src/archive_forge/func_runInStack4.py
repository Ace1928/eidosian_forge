import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def runInStack4(func):
    x = 'inside runInStack4(func)'
    func()
    return x