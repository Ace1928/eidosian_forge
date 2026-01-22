from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def yAutoSpinChanged(self, val):
    self.ctrl[1].autoRadio.setChecked(True)
    self.view().enableAutoRange(ViewBox.YAxis, val * 0.01)