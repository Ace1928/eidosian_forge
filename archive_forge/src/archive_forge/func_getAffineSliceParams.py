import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def getAffineSliceParams(self, data, img, axes=(0, 1), fromBoundingRect=False):
    """
        Returns the parameters needed to use :func:`affineSlice <pyqtgraph.affineSlice>`
        (shape, vectors, origin) to extract a subset of *data* using this ROI 
        and *img* to specify the subset.
        
        If *fromBoundingRect* is True, then the ROI's bounding rectangle is used
        rather than the shape of the ROI.
        
        See :func:`getArrayRegion <pyqtgraph.ROI.getArrayRegion>` for more information.
        """
    if self.scene() is not img.scene():
        raise Exception('ROI and target item must be members of the same scene.')
    origin = img.mapToData(self.mapToItem(img, QtCore.QPointF(0, 0)))
    vx = img.mapToData(self.mapToItem(img, QtCore.QPointF(1, 0))) - origin
    vy = img.mapToData(self.mapToItem(img, QtCore.QPointF(0, 1))) - origin
    lvx = hypot(vx.x(), vx.y())
    lvy = hypot(vy.x(), vy.y())
    sx = 1.0 / lvx
    sy = 1.0 / lvy
    vectors = ((vx.x() * sx, vx.y() * sx), (vy.x() * sy, vy.y() * sy))
    if fromBoundingRect is True:
        shape = (self.boundingRect().width(), self.boundingRect().height())
        origin = img.mapToData(self.mapToItem(img, self.boundingRect().topLeft()))
        origin = (origin.x(), origin.y())
    else:
        shape = self.state['size']
        origin = (origin.x(), origin.y())
    shape = [abs(shape[0] / sx), abs(shape[1] / sy)]
    if img.axisOrder == 'row-major':
        vectors = vectors[::-1]
        shape = shape[::-1]
    return (shape, vectors, origin)