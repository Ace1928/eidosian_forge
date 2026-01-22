from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AtomPairs import Utils
from rdkit.Chem.rdMolDescriptors import (
def pyScorePath(mol, path, size, atomCodes=None):
    """ Returns a score for an individual path.

  >>> from rdkit import Chem
  >>> m = Chem.MolFromSmiles('CCCCC')
  >>> c1 = Utils.GetAtomCode(m.GetAtomWithIdx(0), 1)
  >>> c2 = Utils.GetAtomCode(m.GetAtomWithIdx(1), 2)
  >>> c3 = Utils.GetAtomCode(m.GetAtomWithIdx(2), 2)
  >>> c4 = Utils.GetAtomCode(m.GetAtomWithIdx(3), 1)
  >>> t = c1 | (c2 << rdMolDescriptors.AtomPairsParameters.codeSize) | (c3 << (rdMolDescriptors.AtomPairsParameters.codeSize * 2)) | (c4 << (rdMolDescriptors.AtomPairsParameters.codeSize * 3))
  >>> pyScorePath(m, (0, 1, 2, 3), 4) == t
  1

  The scores are path direction independent:

  >>> pyScorePath(m, (3, 2, 1, 0), 4) == t
  1

  >>> m = Chem.MolFromSmiles('C=CC(=O)O')
  >>> c1 = Utils.GetAtomCode(m.GetAtomWithIdx(0), 1)
  >>> c2 = Utils.GetAtomCode(m.GetAtomWithIdx(1), 2)
  >>> c3 = Utils.GetAtomCode(m.GetAtomWithIdx(2), 2)
  >>> c4 = Utils.GetAtomCode(m.GetAtomWithIdx(4), 1)
  >>> t = c1 | (c2 << rdMolDescriptors.AtomPairsParameters.codeSize) | (c3 << (rdMolDescriptors.AtomPairsParameters.codeSize * 2)) | (c4 << (rdMolDescriptors.AtomPairsParameters.codeSize * 3))
  >>> pyScorePath(m, (0, 1, 2, 4), 4) == t
  1

  """
    codes = [None] * size
    for i in range(size):
        if i == 0 or i == size - 1:
            sub = 1
        else:
            sub = 2
        if not atomCodes:
            codes[i] = Utils.GetAtomCode(mol.GetAtomWithIdx(path[i]), sub)
        else:
            codes[i] = atomCodes[path[i]] - sub
    beg = 0
    end = len(codes) - 1
    while beg < end:
        if codes[beg] == codes[end]:
            beg += 1
            end -= 1
        else:
            if codes[beg] > codes[end]:
                codes.reverse()
            break
    accum = 0
    codeSize = rdMolDescriptors.AtomPairsParameters.codeSize
    for i, code in enumerate(codes):
        accum |= code << codeSize * i
    return accum