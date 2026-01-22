import math
import os.path as op
import pickle
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    if name == 'fpscores':
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict