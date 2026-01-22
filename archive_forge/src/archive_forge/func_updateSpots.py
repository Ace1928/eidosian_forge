import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def updateSpots(self, dataSet=None):
    profiler = debug.Profiler()
    if dataSet is None:
        dataSet = self.data
    invalidate = False
    if self.opts['pxMode'] and self.opts['useCache']:
        mask = dataSet['sourceRect']['w'] == 0
        if np.any(mask):
            invalidate = True
            coords = self.fragmentAtlas[list(zip(*self._style(['symbol', 'size', 'pen', 'brush'], data=dataSet, idx=mask)))]
            dataSet['sourceRect'][mask] = coords
        self._maybeRebuildAtlas()
    else:
        invalidate = True
    self._updateMaxSpotSizes(data=dataSet)
    if invalidate:
        self.invalidate()