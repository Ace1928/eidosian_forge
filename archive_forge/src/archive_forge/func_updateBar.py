from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtWidgets
from .GraphicsObject import GraphicsObject
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .TextItem import TextItem
def updateBar(self):
    view = self.parentItem()
    if view is None:
        return
    p1 = view.mapFromViewToItem(self, QtCore.QPointF(0, 0))
    p2 = view.mapFromViewToItem(self, QtCore.QPointF(self.size, 0))
    w = (p2 - p1).x()
    self.bar.setRect(QtCore.QRectF(-w, 0, w, self._width))
    self.text.setPos(-w / 2.0, 0)