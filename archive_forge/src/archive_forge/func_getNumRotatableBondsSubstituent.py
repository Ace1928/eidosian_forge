import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def getNumRotatableBondsSubstituent(mol, subAtoms):
    rotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    matches = mol.GetSubstructMatches(rotatableBond)
    numRotBonds = 0
    for a1, a2 in matches:
        if a1 in subAtoms and a2 in subAtoms:
            numRotBonds += 1
    return numRotBonds