from PySide2.QtCore import QPoint, QRect, QSize, Qt, qVersion
from PySide2.QtGui import (QBrush, QConicalGradient, QLinearGradient, QPainter,
from PySide2.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
import basicdrawing_rc
def brushChanged(self):
    style = Qt.BrushStyle(self.brushStyleComboBox.itemData(self.brushStyleComboBox.currentIndex(), IdRole))
    if style == Qt.LinearGradientPattern:
        linearGradient = QLinearGradient(0, 0, 100, 100)
        linearGradient.setColorAt(0.0, Qt.white)
        linearGradient.setColorAt(0.2, Qt.green)
        linearGradient.setColorAt(1.0, Qt.black)
        self.renderArea.setBrush(QBrush(linearGradient))
    elif style == Qt.RadialGradientPattern:
        radialGradient = QRadialGradient(50, 50, 50, 70, 70)
        radialGradient.setColorAt(0.0, Qt.white)
        radialGradient.setColorAt(0.2, Qt.green)
        radialGradient.setColorAt(1.0, Qt.black)
        self.renderArea.setBrush(QBrush(radialGradient))
    elif style == Qt.ConicalGradientPattern:
        conicalGradient = QConicalGradient(50, 50, 150)
        conicalGradient.setColorAt(0.0, Qt.white)
        conicalGradient.setColorAt(0.2, Qt.green)
        conicalGradient.setColorAt(1.0, Qt.black)
        self.renderArea.setBrush(QBrush(conicalGradient))
    elif style == Qt.TexturePattern:
        self.renderArea.setBrush(QBrush(QPixmap(':/images/brick.png')))
    else:
        self.renderArea.setBrush(QBrush(Qt.green, style))