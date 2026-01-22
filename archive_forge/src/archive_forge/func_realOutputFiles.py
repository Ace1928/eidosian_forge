import code, sys, traceback
from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkBrush
from .CmdInput import CmdInput
def realOutputFiles(self):
    """Return the real sys.stdout and stderr (which are sometimes masked while running commands)
        """
    return (self._orig_stdout or sys.stdout, self._orig_stderr or sys.stderr)