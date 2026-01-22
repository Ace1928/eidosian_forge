import sys
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
import textures_rc
def freeGLResources(self):
    GLWidget.refCount -= 1
    if GLWidget.refCount == 0:
        self.makeCurrent()
        glDeleteLists(self.__class__.sharedObject, 1)