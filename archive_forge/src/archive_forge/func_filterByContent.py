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
def filterByContent(self, text=None):
    validRegex = True
    try:
        re.compile(text)
        self.ui.exampleFilter.setStyleSheet('')
    except re.error:
        colors = DarkThemeColors if app.property('darkMode') else LightThemeColors
        errorColor = pg.mkColor(colors.Red)
        validRegex = False
        errorColor.setAlpha(100)
        self.ui.exampleFilter.setStyleSheet(f'background: rgba{errorColor.getRgb()}')
    if not validRegex:
        return
    checkDict = unnestedDict(utils.examples_)
    self.hl.searchText = text
    self.hl.setDocument(self.ui.codeView.document())
    titles = []
    text = text.lower()
    for kk, vv in checkDict.items():
        if isinstance(vv, Namespace):
            vv = vv.filename
        filename = os.path.join(path, vv)
        contents = self.getExampleContent(filename).lower()
        if text in contents:
            titles.append(kk)
    self.showExamplesByTitle(titles)