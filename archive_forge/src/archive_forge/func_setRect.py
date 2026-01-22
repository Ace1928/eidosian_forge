import warnings
from collections.abc import Callable
import numpy
from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject
def setRect(self, *args):
    """
        setRect(rect) or setRect(x,y,w,h)
        
        Sets translation and scaling of this ImageItem to display the current image within the rectangle given
        as ``rect`` (:class:`QtCore.QRect` or :class:`QtCore.QRectF`), or described by parameters `x, y, w, h`, 
        defining starting position, width and height.

        This method cannot be used before an image is assigned.
        See the :ref:`examples <ImageItem_examples>` for how to manually set transformations.
        """
    if len(args) == 0:
        self.resetTransform()
        return
    if isinstance(args[0], (QtCore.QRectF, QtCore.QRect)):
        rect = args[0]
    else:
        if hasattr(args[0], '__len__'):
            args = args[0]
        rect = QtCore.QRectF(*args)
    tr = QtGui.QTransform()
    tr.translate(rect.left(), rect.top())
    tr.scale(rect.width() / self.width(), rect.height() / self.height())
    self.setTransform(tr)