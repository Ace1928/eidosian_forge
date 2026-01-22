from ..graphicsItems.GradientEditorItem import GradientEditorItem
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsView import GraphicsView
def setMaxDim(self, mx=None):
    if mx is None:
        mx = self.maxDim
    else:
        self.maxDim = mx
    if self.orientation in ['bottom', 'top']:
        self.setFixedHeight(mx)
        self.setMaximumWidth(16777215)
    else:
        self.setFixedWidth(mx)
        self.setMaximumHeight(16777215)