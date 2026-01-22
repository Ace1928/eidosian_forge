import operator
import weakref
from collections import OrderedDict
from functools import reduce
from math import hypot
from typing import Optional
from xml.etree.ElementTree import Element
from .. import functions as fn
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QtCore, QtWidgets, isQObjectAlive
def childrenShape(self):
    """Return the union of the shapes of all descendants of this item in local coordinates."""
    shapes = [self.mapFromItem(c, c.shape()) for c in self.allChildItems()]
    return reduce(operator.add, shapes)