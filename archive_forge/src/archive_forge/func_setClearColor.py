import sys
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
import textures_rc
def setClearColor(self, color):
    self.clearColor = color
    self.updateGL()