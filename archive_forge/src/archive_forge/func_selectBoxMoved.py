import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def selectBoxMoved(self):
    """The selection box has moved; get its transformation information and pass to the graphics item"""
    self.userTransform = self.selectBox.getGlobalTransform(relativeTo=self.selectBoxBase)
    self.updateTransform()