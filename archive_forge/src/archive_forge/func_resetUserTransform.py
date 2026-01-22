import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def resetUserTransform(self):
    self.userTransform.reset()
    self.updateTransform()
    self.selectBox.blockSignals(True)
    self.selectBoxToItem()
    self.selectBox.blockSignals(False)
    self.sigTransformChanged.emit(self)
    self.sigTransformChangeFinished.emit(self)