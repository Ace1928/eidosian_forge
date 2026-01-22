import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def getLowerBound2D(self, i, j):
    if j > i:
        j, i = (i, j)
    return self._boundsMat2D[i, j]