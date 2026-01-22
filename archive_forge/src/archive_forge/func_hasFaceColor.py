import numpy as np
from ..Qt import QtGui
def hasFaceColor(self):
    """Return True if this data set has face color information"""
    for v in (self._faceColors, self._faceColorsIndexedByFaces, self._faceColorsIndexedByEdges):
        if v is not None:
            return True
    return False