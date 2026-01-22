import copy
from rdkit.Chem.FeatMaps import FeatMaps
def assignMatrix(matrix, i, j, value, constraint):
    if value < constraint:
        matrix[i][j] = value
        matrix[j][i] = value