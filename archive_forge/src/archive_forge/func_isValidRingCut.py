import sys
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import rdqueries
def isValidRingCut(mol):
    """ to check is a fragment is a valid ring cut, it needs to match the
  SMARTS: [$([#0][r].[r][#0]),$([#0][r][#0])] """
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
    return mol.HasSubstructMatch(cSma1) or mol.HasSubstructMatch(cSma2)