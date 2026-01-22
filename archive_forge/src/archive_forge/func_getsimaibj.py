import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def getsimaibj(aipaths, bjpaths, naipaths, nbjpaths):
    """returns the similarity of two sorted path lists.  Equation 2"""
    nc = getcommon(aipaths, naipaths, bjpaths, nbjpaths)
    sim = float(nc + 1) / (max(naipaths, nbjpaths) * 2 - nc + 1)
    return sim