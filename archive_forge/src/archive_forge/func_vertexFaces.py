import numpy as np
from ..Qt import QtGui
def vertexFaces(self):
    """
        Return list mapping each vertex index to a list of face indexes that use the vertex.
        """
    if self._vertexFaces is None:
        self._vertexFaces = [[] for i in range(len(self.vertexes()))]
        for i in range(self._faces.shape[0]):
            face = self._faces[i]
            for ind in face:
                self._vertexFaces[ind].append(i)
    return self._vertexFaces