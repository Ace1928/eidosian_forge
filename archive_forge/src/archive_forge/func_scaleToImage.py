from .. import functions as fn
from .. import getConfigOption
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
def scaleToImage(self, image):
    """Scales such that pixels in image are the same size as screen pixels. This may result in a significant performance increase."""
    pxSize = image.pixelSize()
    image.setPxMode(True)
    try:
        self.sigScaleChanged.disconnect(image.setScaledMode)
    except (TypeError, RuntimeError):
        pass
    tl = image.sceneBoundingRect().topLeft()
    w = self.size().width() * pxSize[0]
    h = self.size().height() * pxSize[1]
    range = QtCore.QRectF(tl.x(), tl.y(), w, h)
    GraphicsView.setRange(self, range, padding=0)
    self.sigScaleChanged.connect(image.setScaledMode)