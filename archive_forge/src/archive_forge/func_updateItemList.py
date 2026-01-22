from .. import exporters as exporters
from .. import functions as fn
from ..graphicsItems.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtWidgets
from . import exportDialogTemplate_generic as ui_template
def updateItemList(self, select=None):
    self.ui.itemTree.clear()
    si = QtWidgets.QTreeWidgetItem(['Entire Scene'])
    si.gitem = self.scene
    self.ui.itemTree.addTopLevelItem(si)
    self.ui.itemTree.setCurrentItem(si)
    si.setExpanded(True)
    for child in self.scene.items():
        if child.parentItem() is None:
            self.updateItemTree(child, si, select=select)