import sys
import numpy as np
from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtWidgets
from .Exporter import Exporter
@staticmethod
def getSupportedImageFormats():
    filter = ['*.' + f.data().decode('utf-8') for f in QtGui.QImageWriter.supportedImageFormats()]
    preferred = ['*.png', '*.tif', '*.jpg']
    for p in preferred[::-1]:
        if p in filter:
            filter.remove(p)
            filter.insert(0, p)
    return filter