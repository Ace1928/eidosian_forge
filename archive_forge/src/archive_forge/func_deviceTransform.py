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
def deviceTransform(self, viewportTransform=None):
    """
        Return the transform that converts local item coordinates to device coordinates (usually pixels).
        Extends deviceTransform to automatically determine the viewportTransform.
        """
    if viewportTransform is None:
        view = self.getViewWidget()
        if view is None:
            return None
        viewportTransform = view.viewportTransform()
    dt = self._qtBaseClass.deviceTransform(self, viewportTransform)
    if dt.determinant() == 0:
        return None
    else:
        return dt