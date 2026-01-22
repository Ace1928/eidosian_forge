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
def updateItem(self):
    self._data['sourceRect'] = (0, 0, 0, 0)
    self._plot.updateSpots(self._data.reshape(1))