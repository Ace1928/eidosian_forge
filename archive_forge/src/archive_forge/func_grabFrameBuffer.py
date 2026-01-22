import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
def grabFrameBuffer(self):
    image = self.glWidget.grabFrameBuffer()
    self.setPixmap(QtGui.QPixmap.fromImage(image))