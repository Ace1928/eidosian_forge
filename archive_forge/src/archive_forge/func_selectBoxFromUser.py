import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def selectBoxFromUser(self):
    """Move the selection box to match the current userTransform"""
    self.selectBox.blockSignals(True)
    self.selectBox.setState(self.selectBoxBase)
    self.selectBox.applyGlobalTransform(self.userTransform)
    self.selectBox.blockSignals(False)