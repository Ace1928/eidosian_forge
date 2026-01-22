import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def getsimmatrix(m1, m1pathintegers, m2, m2pathintegers):
    """generate a matrix of atom atom similarities.  See Figure 4"""
    aidata = [((ai.GetAtomicNum(), ai.GetIsAromatic()), ai.GetIdx()) for ai in m1.GetAtoms()]
    bjdata = [((bj.GetAtomicNum(), bj.GetIsAromatic()), bj.GetIdx()) for bj in m2.GetAtoms()]
    simmatrixarray = numpy.zeros((len(aidata), len(bjdata)))
    for ai, (aitype, aiidx) in enumerate(aidata):
        aipaths = m1pathintegers[aiidx]
        naipaths = len(aipaths)
        for bj, (bjtype, bjidx) in enumerate(bjdata):
            if aitype == bjtype:
                bjpaths = m2pathintegers[bjidx]
                nbjpaths = len(bjpaths)
                simmatrixarray[ai][bj] = getsimaibj(aipaths, bjpaths, naipaths, nbjpaths)
    return simmatrixarray