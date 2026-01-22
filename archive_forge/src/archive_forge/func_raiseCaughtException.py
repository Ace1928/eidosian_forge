import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def raiseCaughtException():
    """Raise and catch an exception
    """
    x = 'inside raiseCaughtException()'
    try:
        raise Exception(f'Raised an exception {x} in {threadName()}')
    except Exception:
        print(f'Raised and caught exception {x} in {threadName()}  trace: {sys._getframe().f_trace}')