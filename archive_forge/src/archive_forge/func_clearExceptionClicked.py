import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def clearExceptionClicked(self):
    self.stackTree.clear()
    self.clearExceptionBtn.setEnabled(False)