from PySide2.QtCore import (Signal, QMutex, QMutexLocker, QPoint, QSize, Qt,
from PySide2.QtGui import QColor, QImage, QPainter, QPixmap, qRgb
from PySide2.QtWidgets import QApplication, QWidget
PySide2 port of the corelib/threads/mandelbrot example from Qt v5.x, originating from PyQt