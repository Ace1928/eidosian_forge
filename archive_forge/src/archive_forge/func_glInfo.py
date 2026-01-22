from argparse import ArgumentParser, RawTextHelpFormatter
import numpy
import sys
from textwrap import dedent
from PySide2.QtCore import QCoreApplication, QLibraryInfo, QSize, QTimer, Qt
from PySide2.QtGui import (QMatrix4x4, QOpenGLBuffer, QOpenGLContext, QOpenGLShader,
from PySide2.QtWidgets import (QApplication, QHBoxLayout, QMessageBox, QPlainTextEdit,
from PySide2.support import VoidPtr
def glInfo(self):
    if not self.context.makeCurrent(self):
        raise Exception('makeCurrent() failed')
    functions = self.context.functions()
    text = 'Vendor: {}\nRenderer: {}\nVersion: {}\nShading language: {}\n\nContext Format: {}\n\nSurface Format: {}'.format(functions.glGetString(GL.GL_VENDOR), functions.glGetString(GL.GL_RENDERER), functions.glGetString(GL.GL_VERSION), functions.glGetString(GL.GL_SHADING_LANGUAGE_VERSION), print_surface_format(self.context.format()), print_surface_format(self.format()))
    self.context.doneCurrent()
    return text