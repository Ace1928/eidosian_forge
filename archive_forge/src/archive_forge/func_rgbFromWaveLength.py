from PySide2.QtCore import (Signal, QMutex, QMutexLocker, QPoint, QSize, Qt,
from PySide2.QtGui import QColor, QImage, QPainter, QPixmap, qRgb
from PySide2.QtWidgets import QApplication, QWidget
def rgbFromWaveLength(self, wave):
    r = 0.0
    g = 0.0
    b = 0.0
    if wave >= 380.0 and wave <= 440.0:
        r = -1.0 * (wave - 440.0) / (440.0 - 380.0)
        b = 1.0
    elif wave >= 440.0 and wave <= 490.0:
        g = (wave - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif wave >= 490.0 and wave <= 510.0:
        g = 1.0
        b = -1.0 * (wave - 510.0) / (510.0 - 490.0)
    elif wave >= 510.0 and wave <= 580.0:
        r = (wave - 510.0) / (580.0 - 510.0)
        g = 1.0
    elif wave >= 580.0 and wave <= 645.0:
        r = 1.0
        g = -1.0 * (wave - 645.0) / (645.0 - 580.0)
    elif wave >= 645.0 and wave <= 780.0:
        r = 1.0
    s = 1.0
    if wave > 700.0:
        s = 0.3 + 0.7 * (780.0 - wave) / (780.0 - 700.0)
    elif wave < 420.0:
        s = 0.3 + 0.7 * (wave - 380.0) / (420.0 - 380.0)
    r = pow(r * s, 0.8)
    g = pow(g * s, 0.8)
    b = pow(b * s, 0.8)
    return qRgb(r * 255, g * 255, b * 255)