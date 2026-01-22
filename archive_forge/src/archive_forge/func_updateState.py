from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def updateState(self):
    state = self.view().getState(copy=False)
    if state['mouseMode'] == ViewBox.PanMode:
        self.mouseModes[0].setChecked(True)
    else:
        self.mouseModes[1].setChecked(True)
    for i in [0, 1]:
        tr = state['targetRange'][i]
        self.ctrl[i].minText.setText('%0.5g' % tr[0])
        self.ctrl[i].maxText.setText('%0.5g' % tr[1])
        if state['autoRange'][i] is not False:
            self.ctrl[i].autoRadio.setChecked(True)
            if state['autoRange'][i] is not True:
                self.ctrl[i].autoPercentSpin.setValue(int(state['autoRange'][i] * 100))
        else:
            self.ctrl[i].manualRadio.setChecked(True)
        self.ctrl[i].mouseCheck.setChecked(state['mouseEnabled'][i])
        c = self.ctrl[i].linkCombo
        c.blockSignals(True)
        try:
            view = state['linkedViews'][i]
            if view is None:
                view = ''
            ind = c.findText(view)
            if ind == -1:
                ind = 0
            c.setCurrentIndex(ind)
        finally:
            c.blockSignals(False)
        self.ctrl[i].autoPanCheck.setChecked(state['autoPan'][i])
        self.ctrl[i].visibleOnlyCheck.setChecked(state['autoVisibleOnly'][i])
        xy = ['x', 'y'][i]
        self.ctrl[i].invertCheck.setChecked(state.get(xy + 'Inverted', False))
    self.valid = True