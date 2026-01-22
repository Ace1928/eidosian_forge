import os
import sys
import pickle
import subprocess
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets
from .repl_widget import ReplWidget
from .exception_widget import ExceptionHandlerWidget
def loadHistory(self):
    """Return the list of previously-invoked command strings (or None)."""
    if self.historyFile is not None and os.path.exists(self.historyFile):
        with open(self.historyFile, 'rb') as pf:
            return pickle.load(pf)