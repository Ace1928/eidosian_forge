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
def showExamplesByTitle(self, titles):
    QTWI = QtWidgets.QTreeWidgetItemIterator
    flag = QTWI.IteratorFlag.NoChildren
    treeIter = QTWI(self.ui.exampleTree, flag)
    item = treeIter.value()
    while item is not None:
        parent = item.parent()
        show = item.childCount() or item.text(0) in titles
        item.setHidden(not show)
        if parent:
            hideParent = True
            for ii in range(parent.childCount()):
                if not parent.child(ii).isHidden():
                    hideParent = False
                    break
            parent.setHidden(hideParent)
        treeIter += 1
        item = treeIter.value()