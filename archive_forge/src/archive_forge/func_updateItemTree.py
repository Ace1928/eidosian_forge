from .. import exporters as exporters
from .. import functions as fn
from ..graphicsItems.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtWidgets
from . import exportDialogTemplate_generic as ui_template
def updateItemTree(self, item, treeItem, select=None):
    si = None
    if isinstance(item, ViewBox):
        si = QtWidgets.QTreeWidgetItem(['ViewBox'])
    elif isinstance(item, PlotItem):
        si = QtWidgets.QTreeWidgetItem(['Plot'])
    if si is not None:
        si.gitem = item
        treeItem.addChild(si)
        treeItem = si
        if si.gitem is select:
            self.ui.itemTree.setCurrentItem(si)
    for ch in item.childItems():
        self.updateItemTree(ch, treeItem, select=select)