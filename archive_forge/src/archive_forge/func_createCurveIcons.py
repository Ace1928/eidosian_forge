from PySide2 import QtCore, QtGui, QtWidgets
import easing_rc
from ui_form import Ui_Form
def createCurveIcons(self):
    pix = QtGui.QPixmap(self.m_iconSize)
    painter = QtGui.QPainter()
    gradient = QtGui.QLinearGradient(0, 0, 0, self.m_iconSize.height())
    gradient.setColorAt(0.0, QtGui.QColor(240, 240, 240))
    gradient.setColorAt(1.0, QtGui.QColor(224, 224, 224))
    brush = QtGui.QBrush(gradient)
    curve_types = [(n, c) for n, c in QtCore.QEasingCurve.__dict__.items() if isinstance(c, QtCore.QEasingCurve.Type) and c != QtCore.QEasingCurve.Custom and (c != QtCore.QEasingCurve.NCurveTypes) and (c != QtCore.QEasingCurve.TCBSpline)]
    curve_types.sort(key=lambda ct: ct[1])
    painter.begin(pix)
    for curve_name, curve_type in curve_types:
        painter.fillRect(QtCore.QRect(QtCore.QPoint(0, 0), self.m_iconSize), brush)
        curve = QtCore.QEasingCurve(curve_type)
        painter.setPen(QtGui.QColor(0, 0, 255, 64))
        xAxis = self.m_iconSize.height() / 1.5
        yAxis = self.m_iconSize.width() / 3.0
        painter.drawLine(0, xAxis, self.m_iconSize.width(), xAxis)
        painter.drawLine(yAxis, 0, yAxis, self.m_iconSize.height())
        curveScale = self.m_iconSize.height() / 2.0
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.red)
        start = QtCore.QPoint(yAxis, xAxis - curveScale * curve.valueForProgress(0))
        painter.drawRect(start.x() - 1, start.y() - 1, 3, 3)
        painter.setBrush(QtCore.Qt.blue)
        end = QtCore.QPoint(yAxis + curveScale, xAxis - curveScale * curve.valueForProgress(1))
        painter.drawRect(end.x() - 1, end.y() - 1, 3, 3)
        curvePath = QtGui.QPainterPath()
        curvePath.moveTo(QtCore.QPointF(start))
        t = 0.0
        while t <= 1.0:
            to = QtCore.QPointF(yAxis + curveScale * t, xAxis - curveScale * curve.valueForProgress(t))
            curvePath.lineTo(to)
            t += 1.0 / curveScale
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.strokePath(curvePath, QtGui.QColor(32, 32, 32))
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        item = QtWidgets.QListWidgetItem()
        item.setIcon(QtGui.QIcon(pix))
        item.setText(curve_name)
        self.m_ui.easingCurvePicker.addItem(item)
    painter.end()