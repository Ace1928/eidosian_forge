import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def freeResources(self):
    self.makeCurrent()
    GL.glDeleteLists(self.object, 1)