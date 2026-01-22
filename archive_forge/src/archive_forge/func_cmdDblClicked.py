import os
import sys
import pickle
import subprocess
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets
from .repl_widget import ReplWidget
from .exception_widget import ExceptionHandlerWidget
def cmdDblClicked(self, item):
    index = -(self.historyList.row(item) + 1)
    self.input.setHistory(index)
    self.input.execCmd()