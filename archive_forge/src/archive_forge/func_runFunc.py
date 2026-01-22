import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def runFunc(func):
    if signalCheck.isChecked():
        if queuedSignalCheck.isChecked():
            func = functools.partial(queuedSignalEmitter.signal.emit, runInStack, (func,))
        else:
            func = functools.partial(signalEmitter.signal.emit, runInStack, (func,))
    if threadCheck.isChecked():
        threadRunQueue.put((runInStack, (func,)))
    else:
        runInStack(func)