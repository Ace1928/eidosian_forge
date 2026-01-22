from .. import functions as fn
from .. import getConfigOption
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
def setCentralWidget(self, item):
    """Sets a QGraphicsWidget to automatically fill the entire view (the item will be automatically
        resize whenever the GraphicsView is resized)."""
    if self.centralWidget is not None:
        self.scene().removeItem(self.centralWidget)
    self.centralWidget = item
    if item is not None:
        self.sceneObj.addItem(item)
        self.resizeEvent(None)