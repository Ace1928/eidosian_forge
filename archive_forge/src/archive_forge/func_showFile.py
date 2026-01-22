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
def showFile(self):
    fn = self.currentFile()
    text = self.getExampleContent(fn)
    self.ui.codeView.setPlainText(text)
    self.ui.loadedFileLabel.setText(fn)
    self.codeBtn.hide()