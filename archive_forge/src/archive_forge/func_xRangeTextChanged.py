from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def xRangeTextChanged(self):
    self.ctrl[0].manualRadio.setChecked(True)
    self.view().setXRange(*self._validateRangeText(0), padding=0)