from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def getRoiData(self):
    d = self.roi.getArrayRegion(self.roiData, self.roiImg, axes=self.axes)
    if d is None:
        return
    while d.ndim > 1:
        d = d.mean(axis=1)
    return d