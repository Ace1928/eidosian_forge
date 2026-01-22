from argparse import ArgumentParser, RawTextHelpFormatter
import numpy
import sys
from textwrap import dedent
from PySide2.QtCore import QCoreApplication, QLibraryInfo, QSize, QTimer, Qt
from PySide2.QtGui import (QMatrix4x4, QOpenGLBuffer, QOpenGLContext, QOpenGLShader,
from PySide2.QtWidgets import (QApplication, QHBoxLayout, QMessageBox, QPlainTextEdit,
from PySide2.support import VoidPtr
def initGl(self):
    self.program = QOpenGLShaderProgram(self)
    self.vao = QOpenGLVertexArrayObject()
    self.vbo = QOpenGLBuffer()
    format = self.context.format()
    useNewStyleShader = format.profile() == QSurfaceFormat.CoreProfile
    if format.renderableType() == QSurfaceFormat.OpenGL and format.majorVersion() == 3 and (format.minorVersion() <= 1):
        useNewStyleShader = not format.testOption(QSurfaceFormat.DeprecatedFunctions)
    vertexShader = vertexShaderSource if useNewStyleShader else vertexShaderSource110
    fragmentShader = fragmentShaderSource if useNewStyleShader else fragmentShaderSource110
    if not self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertexShader):
        raise Exception('Vertex shader could not be added: {} ({})'.format(self.program.log(), vertexShader))
    if not self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragmentShader):
        raise Exception('Fragment shader could not be added: {} ({})'.format(self.program.log(), fragmentShader))
    if not self.program.link():
        raise Exception('Could not link shaders: {}'.format(self.program.log()))
    self.posAttr = self.program.attributeLocation('posAttr')
    self.colAttr = self.program.attributeLocation('colAttr')
    self.matrixUniform = self.program.uniformLocation('matrix')
    self.vbo.create()
    self.vbo.bind()
    self.verticesData = vertices.tobytes()
    self.colorsData = colors.tobytes()
    verticesSize = 4 * vertices.size
    colorsSize = 4 * colors.size
    self.vbo.allocate(VoidPtr(self.verticesData), verticesSize + colorsSize)
    self.vbo.write(verticesSize, VoidPtr(self.colorsData), colorsSize)
    self.vbo.release()
    vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)
    if self.vao.isCreated():
        self.setupVertexAttribs()