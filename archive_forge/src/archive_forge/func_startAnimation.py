from PySide2 import QtCore, QtGui, QtWidgets
import easing_rc
from ui_form import Ui_Form
def startAnimation(self):
    self.m_anim.setStartValue(QtCore.QPointF(0, 0))
    self.m_anim.setEndValue(QtCore.QPointF(100, 100))
    self.m_anim.setDuration(2000)
    self.m_anim.setLoopCount(-1)
    self.m_anim.start()