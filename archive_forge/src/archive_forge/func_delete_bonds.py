import sys
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import rdqueries
def delete_bonds(mol, bonds, ftype, hac):
    """ Fragment molecule on bonds and reduce to fraggle fragmentation SMILES.
  If none exists, returns None """
    bondIdx = [mol.GetBondBetweenAtoms(*bond).GetIdx() for bond in bonds]
    modifiedMol = Chem.FragmentOnBonds(mol, bondIdx, dummyLabels=[(0, 0)] * len(bondIdx))
    Chem.SanitizeMol(modifiedMol, Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
    fragments = Chem.GetMolFrags(modifiedMol, asMols=True, sanitizeFrags=False)
    return select_fragments(fragments, ftype, hac)