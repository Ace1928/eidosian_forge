import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def calcSizeSubstituents(mol, subs, sharedNeighbors, maxShell):
    sizeDict = defaultdict()
    numAtoms = mol.GetNumAtoms()
    for sidx, sub in sorted(subs.items(), key=lambda x: len(x[1])):
        size = _getSizeOfSubstituents(sub, sharedNeighbors)
        numNOs = 0
        numShared = 0
        for i in sub:
            if mol.GetAtomWithIdx(i).GetAtomicNum() in [7, 8]:
                numNOs += 1.0 / sharedNeighbors[i]
            if sharedNeighbors[i] > 1:
                numShared += 1
        numRotBs = getNumRotatableBondsSubstituent(mol, set(sub))
        aroBonds = getNumAromaticBondsSubstituent(mol, set(sub))
        sizeDict[sidx] = substituentDescriptor(size=size, relSize=size / numAtoms, numNO=numNOs, relNumNO=numNOs / numAtoms, relNumNO_2=numNOs / size, pathLength=maxShell[sidx], relPathLength=maxShell[sidx] / numAtoms, relPathLength_2=maxShell[sidx] / size, sharedNeighbors=numShared, numRotBonds=numRotBs, numAroBonds=aroBonds)
    if len(sizeDict) < 4:
        for i in range(4 - len(sizeDict)):
            sizeDict['H' + str(i)] = substituentDescriptor(size=0, relSize=0, numNO=0, relNumNO=0, relNumNO_2=0, pathLength=0, relPathLength=0, relPathLength_2=0, sharedNeighbors=0, numRotBonds=0, numAroBonds=0)
    return sizeDict