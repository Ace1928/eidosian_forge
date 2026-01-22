import bisect
import numpy
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors, rdPartialCharges
def pyLabuteASA(mol, includeHs=1):
    """ calculates Labute's Approximate Surface Area (ASA from MOE)

    Definition from P. Labute's article in the Journal of the Chemical Computing Group
    and J. Mol. Graph. Mod.  _18_ 464-477 (2000)

  """
    Vi = _LabuteHelper(mol, includeHs=includeHs)
    return sum(Vi)