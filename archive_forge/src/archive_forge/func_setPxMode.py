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
def setPxMode(self, mode):
    if self.opts['pxMode'] == mode:
        return
    self.opts['pxMode'] = mode
    self.invalidate()