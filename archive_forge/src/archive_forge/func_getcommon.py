import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def getcommon(l1, ll1, l2, ll2):
    """returns the number of items sorted lists l1 and l2 have in common.  ll1 and ll2 are the list lengths"""
    ncommon = 0
    ix1 = 0
    ix2 = 0
    while ix1 < ll1 and ix2 < ll2:
        a1 = l1[ix1]
        a2 = l2[ix2]
        if a1 < a2:
            ix1 += 1
        elif a1 > a2:
            ix2 += 1
        else:
            ncommon += 1
            ix1 += 1
            ix2 += 1
    return ncommon