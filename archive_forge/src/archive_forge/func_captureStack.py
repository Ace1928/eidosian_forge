import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def captureStack():
    """Inspect the curent call stack
    """
    x = 'inside captureStack()'
    global console
    console.setStack()
    return x