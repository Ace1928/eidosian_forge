import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def replaceHandle(self, oldHandle, newHandle):
    """Replace one handle in the ROI for another. This is useful when 
        connecting multiple ROIs together.
        
        *oldHandle* may be a Handle instance or the index of a handle to be
        replaced."""
    index = self.indexOfHandle(oldHandle)
    info = self.handles[index]
    self.removeHandle(index)
    info['item'] = newHandle
    info['pos'] = newHandle.pos()
    self.addHandle(info, index=index)