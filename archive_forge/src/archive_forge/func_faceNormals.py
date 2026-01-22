import numpy as np
from ..Qt import QtGui
def faceNormals(self, indexed=None):
    """
        Return an array (Nf, 3) of normal vectors for each face.
        If indexed='faces', then instead return an indexed array
        (Nf, 3, 3)  (this is just the same array with each vector
        copied three times).
        """
    if self._faceNormals is None:
        v = self.vertexes(indexed='faces')
        self._faceNormals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    if indexed is None:
        return self._faceNormals
    elif indexed == 'faces':
        if self._faceNormalsIndexedByFaces is None:
            norms = np.empty((self._faceNormals.shape[0], 3, 3), dtype=np.float32)
            norms[:] = self._faceNormals[:, np.newaxis, :]
            self._faceNormalsIndexedByFaces = norms
        return self._faceNormalsIndexedByFaces
    else:
        raise Exception("Invalid indexing mode. Accepts: None, 'faces'")