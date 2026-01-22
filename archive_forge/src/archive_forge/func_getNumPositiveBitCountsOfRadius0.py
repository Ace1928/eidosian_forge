import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
def getNumPositiveBitCountsOfRadius0(fp, bitinfo):
    count = 0
    bitsUnmappedAtoms = []
    for k in bitinfo:
        if bitinfo[k][0][1] == 0:
            v = fp[k]
            if v > 0:
                count += 1
                bitsUnmappedAtoms.append((k, v))
    return (count, bitsUnmappedAtoms)