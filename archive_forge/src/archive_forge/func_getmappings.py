import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def getmappings(simmatrixarray):
    """return a mapping of the atoms in the similarity matix using the heuristic algorithm described in the paper"""
    costarray = numpy.ones(simmatrixarray.shape) - simmatrixarray
    it = numpy.nditer(costarray, flags=['multi_index'], op_flags=['writeonly'])
    dsu = []
    for a in it:
        dsu.append((a, it.multi_index[0], it.multi_index[1]))
    dsu.sort()
    seena = set()
    seenb = set()
    mappings = []
    for sim, a, b in dsu:
        if a not in seena and b not in seenb:
            seena.add(a)
            seenb.add(b)
            mappings.append((a, b))
    return mappings[:min(simmatrixarray.shape)]