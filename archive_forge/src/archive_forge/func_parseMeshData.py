from OpenGL.GL import *  # noqa
import numpy as np
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
from ..MeshData import MeshData
def parseMeshData(self):
    if self.vertexes is not None and self.normals is not None:
        return
    if self.opts['meshdata'] is not None:
        md = self.opts['meshdata']
        if self.opts['smooth'] and (not md.hasFaceIndexedData()):
            self.vertexes = md.vertexes()
            if self.opts['computeNormals']:
                self.normals = md.vertexNormals()
            self.faces = md.faces()
            if md.hasVertexColor():
                self.colors = md.vertexColors()
            if md.hasFaceColor():
                self.colors = md.faceColors()
        else:
            self.vertexes = md.vertexes(indexed='faces')
            if self.opts['computeNormals']:
                if self.opts['smooth']:
                    self.normals = md.vertexNormals(indexed='faces')
                else:
                    self.normals = md.faceNormals(indexed='faces')
            self.faces = None
            if md.hasVertexColor():
                self.colors = md.vertexColors(indexed='faces')
            elif md.hasFaceColor():
                self.colors = md.faceColors(indexed='faces')
        if self.opts['drawEdges']:
            if not md.hasFaceIndexedData():
                self.edges = md.edges()
                self.edgeVerts = md.vertexes()
            else:
                self.edges = md.edges()
                self.edgeVerts = md.vertexes(indexed='faces')
        return