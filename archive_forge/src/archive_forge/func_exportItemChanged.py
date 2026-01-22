from .. import exporters as exporters
from .. import functions as fn
from ..graphicsItems.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtWidgets
from . import exportDialogTemplate_generic as ui_template
def exportItemChanged(self, item, prev):
    if item is None:
        return
    if item.gitem is self.scene:
        newBounds = self.scene.views()[0].viewRect()
    else:
        newBounds = item.gitem.sceneBoundingRect()
    self.selectBox.setRect(newBounds)
    self.selectBox.show()
    self.updateFormatList()