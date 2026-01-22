import os
import re
import subprocess
import sys
import time
import warnings
from . import QtCore, QtGui, QtWidgets, compat
from . import internals
def pyqt_qabort_override(*args, **kwds):
    return sys_excepthook(*args, **kwds)