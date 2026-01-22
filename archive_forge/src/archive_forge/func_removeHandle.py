import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def removeHandle(self, handle, updateSegments=True):
    ROI.removeHandle(self, handle)
    handle.sigRemoveRequested.disconnect(self.removeHandle)
    if not updateSegments:
        return
    segments = handle.rois[:]
    if len(segments) == 1:
        self.removeSegment(segments[0])
    elif len(segments) > 1:
        handles = [h['item'] for h in segments[1].handles]
        handles.remove(handle)
        segments[0].replaceHandle(handle, handles[0])
        self.removeSegment(segments[1])
    self.stateChanged(finish=True)