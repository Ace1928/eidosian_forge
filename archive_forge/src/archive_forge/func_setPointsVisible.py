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
def setPointsVisible(self, visible, update=True, dataSet=None, mask=None):
    """Set whether or not each spot is visible.
        If a list or array is provided, then the visibility for each spot will be set separately.
        Otherwise, the argument will be used for all spots."""
    if dataSet is None:
        dataSet = self.data
    if isinstance(visible, np.ndarray) or isinstance(visible, list):
        visibilities = visible
        if mask is not None:
            visibilities = visibilities[mask]
        if len(visibilities) != len(dataSet):
            raise Exception('Number of visibilities does not match number of points (%d != %d)' % (len(visibilities), len(dataSet)))
        dataSet['visible'] = visibilities
    else:
        dataSet['visible'] = visible
    dataSet['sourceRect'] = 0
    if update:
        self.updateSpots(dataSet)