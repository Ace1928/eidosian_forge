import keyword
import os
import pkgutil
import re
import subprocess
import sys
from argparse import Namespace
from collections import OrderedDict
from functools import lru_cache
import pyqtgraph as pg
from pyqtgraph.Qt import QT_LIB, QtCore, QtGui, QtWidgets
import exampleLoaderTemplate_generic as ui_template
import utils
@lru_cache(100)
def getExampleContent(self, filename):
    if filename is None:
        self.ui.codeView.clear()
        return
    if os.path.isdir(filename):
        filename = os.path.join(filename, '__main__.py')
    with open(filename, 'r') as currentFile:
        text = currentFile.read()
    return text