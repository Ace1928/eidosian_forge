import os
import re
import subprocess
import sys
import time
import warnings
from . import QtCore, QtGui, QtWidgets, compat
from . import internals
@staticmethod
def qWait(msec):
    start = time.time()
    QtWidgets.QApplication.processEvents()
    while time.time() < start + msec * 0.001:
        QtWidgets.QApplication.processEvents()