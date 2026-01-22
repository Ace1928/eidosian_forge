from PySide2 import QtCore, QtGui, QtWidgets
import easing_rc
from ui_form import Ui_Form
def curveChanged(self, row):
    curveType = QtCore.QEasingCurve.Type(row)
    self.m_anim.setEasingCurve(curveType)
    self.m_anim.setCurrentTime(0)
    isElastic = curveType >= QtCore.QEasingCurve.InElastic and curveType <= QtCore.QEasingCurve.OutInElastic
    isBounce = curveType >= QtCore.QEasingCurve.InBounce and curveType <= QtCore.QEasingCurve.OutInBounce
    self.m_ui.periodSpinBox.setEnabled(isElastic)
    self.m_ui.amplitudeSpinBox.setEnabled(isElastic or isBounce)
    self.m_ui.overshootSpinBox.setEnabled(curveType >= QtCore.QEasingCurve.InBack and curveType <= QtCore.QEasingCurve.OutInBack)