import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
def setPixmap(self, pixmap):
    self.pixmapLabel.setPixmap(pixmap)
    size = pixmap.size()
    if size - QtCore.QSize(1, 0) == self.pixmapLabelArea.maximumViewportSize():
        size -= QtCore.QSize(1, 0)
    self.pixmapLabel.resize(size)