import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def showSelectBox(self):
    """Display the selection box around this item if it is selected and movable"""
    if self.selectedAlone and self.isMovable() and self.isVisible():
        self.selectBox.show()
    else:
        self.selectBox.hide()