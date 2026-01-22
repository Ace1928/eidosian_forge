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
def redirectItems(self, canvas):
    for i in self.items:
        if i is self.grid:
            continue
        li = i.listItem
        parent = li.parent()
        if parent is None:
            tree = li.treeWidget()
            if tree is None:
                print('Skipping item', i, i.name)
                continue
            tree.removeTopLevelItem(li)
        else:
            parent.removeChild(li)
        canvas.addItem(i)