from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def xLinkComboChanged(self, ind):
    self.view().setXLink(str(self.ctrl[0].linkCombo.currentText()))