import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def paintGL(self):
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glLoadIdentity()
    GL.glTranslated(0.0, 0.0, -10.0)
    GL.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
    GL.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
    GL.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)
    GL.glCallList(self.object)