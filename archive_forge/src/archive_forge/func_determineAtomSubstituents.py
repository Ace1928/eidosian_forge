import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def determineAtomSubstituents(atomID, mol, distanceMatrix, verbose=False):
    atomPaths = distanceMatrix[atomID]
    neighbors = [n for n, i in enumerate(atomPaths) if i == 1]
    subs = defaultdict(list)
    sharedNeighbors = defaultdict(int)
    maxShell = defaultdict(int)
    for n in neighbors:
        subs[n].append(n)
        sharedNeighbors[n] += 1
        maxShell[n] = 0
    mindist = 2
    maxdist = int(np.max(atomPaths))
    for d in range(mindist, maxdist + 1):
        if verbose:
            print('Shell: ', d)
        newShell = [n for n, i in enumerate(atomPaths) if i == d]
        for aidx in newShell:
            if verbose:
                print('Atom ', aidx, ' in shell ', d)
            atom = mol.GetAtomWithIdx(aidx)
            for n in atom.GetNeighbors():
                nidx = n.GetIdx()
                for k, v in subs.items():
                    if nidx in v and nidx not in newShell and (aidx not in v):
                        subs[k].append(aidx)
                        sharedNeighbors[aidx] += 1
                        maxShell[k] = d
                        if verbose:
                            print('Atom ', aidx, ' assigned to ', nidx)
    if verbose:
        print(subs)
        print(sharedNeighbors)
    return (subs, sharedNeighbors, maxShell)