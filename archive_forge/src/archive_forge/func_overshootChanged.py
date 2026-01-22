from PySide2 import QtCore, QtGui, QtWidgets
import easing_rc
from ui_form import Ui_Form
def overshootChanged(self, value):
    curve = self.m_anim.easingCurve()
    curve.setOvershoot(value)
    self.m_anim.setEasingCurve(curve)