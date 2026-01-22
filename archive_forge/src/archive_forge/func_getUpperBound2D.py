import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def getUpperBound2D(self, i, j):
    if i > j:
        j, i = (i, j)
    return self._boundsMat2D[i, j]