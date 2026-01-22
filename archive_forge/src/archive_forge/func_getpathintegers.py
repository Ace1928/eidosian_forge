import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def getpathintegers(m1, uptolength=7):
    """returns a list of integers describing the paths for molecule m1.  This uses numpy 16 bit unsigned integers to reproduce the data in the Gobbi paper.  The returned list is sorted"""
    bondtypelookup = {}
    for b in m1.GetBonds():
        bondtypelookup[b.GetIdx()] = (_BK_[b.GetBondType()], b.GetBeginAtom(), b.GetEndAtom())
    pathintegers = {}
    for a in m1.GetAtoms():
        idx = a.GetIdx()
        pathintegers[idx] = []
        for ipath, path in enumerate(FindAllPathsOfLengthMToN_Gobbi(m1, 1, uptolength, rootedAtAtom=idx, uniquepaths=False)):
            strpath = []
            currentidx = idx
            res = []
            for ip, p in enumerate(path):
                bk, a1, a2 = bondtypelookup[p]
                strpath.append(_BONDSYMBOL_[bk])
                if a1.GetIdx() == currentidx:
                    a = a2
                else:
                    a = a1
                ak = a.GetAtomicNum()
                if a.GetIsAromatic():
                    ak += 108
                if a.GetIdx() == idx:
                    ak = None
                if ak is not None:
                    astr = a.GetSymbol()
                    if a.GetIsAromatic():
                        strpath.append(astr.lower())
                    else:
                        strpath.append(astr)
                res.append((bk, ak))
                currentidx = a.GetIdx()
            pathuniqueint = numpy.ushort(0)
            for ires, (bi, ai) in enumerate(res):
                val1 = pathuniqueint + numpy.ushort(bi)
                val2 = val1 * numpy.ushort(_nAT_)
                if ai is not None:
                    val3 = val2 + numpy.ushort(ai)
                    val4 = val3 * numpy.ushort(_nBT_)
                else:
                    val4 = val2
                pathuniqueint = val4
            pathintegers[idx].append(pathuniqueint)
    for p in pathintegers.values():
        p.sort()
    return pathintegers