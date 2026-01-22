import sys
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import rdqueries
def partialSimilarity(atomID):
    """ Determine similarity for the atoms set by atomID """
    modifiedFP = DataStructs.ExplicitBitVect(1024)
    modifiedFP.SetBitsFromList(aBits[atomID])
    return DataStructs.TverskySimilarity(subsFp, modifiedFP, 0, 1)