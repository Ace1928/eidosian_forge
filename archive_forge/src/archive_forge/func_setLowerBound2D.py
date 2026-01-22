import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def setLowerBound2D(self, i, j, val, checkBounds=False):
    if checkBounds:
        self._checkBounds(i, j)
    if j > i:
        j, i = (i, j)
    self._boundsMat2D[i, j] = val