import os
import signal
import threading
from pydev_ipython.qt_for_kernel import QtCore, QtGui
from pydev_ipython.inputhook import allow_CTRL_C, ignore_CTRL_C, stdin_ready
def set_hook(self, *args, **kwargs):
    pass