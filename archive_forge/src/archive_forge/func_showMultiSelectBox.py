import gc
import importlib
import weakref
import warnings
from ..graphicsItems.GridItem import GridItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
from . import CanvasTemplate_generic as ui_template
from .CanvasItem import CanvasItem, GroupCanvasItem
from .CanvasManager import CanvasManager
def showMultiSelectBox(self):
    items = self.selectedItems()
    rect = self.view.itemBoundingRect(items[0].graphicsItem())
    for i in items:
        if not i.isMovable():
            return
        br = self.view.itemBoundingRect(i.graphicsItem())
        rect = rect | br
    self.multiSelectBox.blockSignals(True)
    self.multiSelectBox.setPos([rect.x(), rect.y()])
    self.multiSelectBox.setSize(rect.size())
    self.multiSelectBox.setAngle(0)
    self.multiSelectBox.blockSignals(False)
    self.multiSelectBox.show()
    self.ui.mirrorSelectionBtn.show()
    self.ui.reflectSelectionBtn.show()
    self.ui.resetTransformsBtn.show()