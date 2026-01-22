import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def threadRunner():
    global threadRunQueue
    sys.settrace(lambda *args: None)
    while True:
        func, args = threadRunQueue.get()
        try:
            print(f'running {func} from thread, trace: {sys._getframe().f_trace}')
            func(*args)
        except Exception:
            sys.excepthook(*sys.exc_info())