from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets, QtGui
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
def itemRect(self):
    return self.item.mapRectToParent(self.item.boundingRect())