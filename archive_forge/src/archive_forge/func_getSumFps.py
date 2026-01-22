import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
def getSumFps(fps):
    summedFP = copy.deepcopy(fps[0])
    for fp in fps[1:]:
        summedFP += fp
    return summedFP