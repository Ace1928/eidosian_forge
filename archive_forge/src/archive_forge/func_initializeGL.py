import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def initializeGL(self):
    darkTrolltechPurple = self.trolltechPurple.darker()
    GL.glClearColor(darkTrolltechPurple.redF(), darkTrolltechPurple.greenF(), darkTrolltechPurple.blueF(), darkTrolltechPurple.alphaF())
    self.object = self.makeObject()
    GL.glShadeModel(GL.GL_FLAT)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)