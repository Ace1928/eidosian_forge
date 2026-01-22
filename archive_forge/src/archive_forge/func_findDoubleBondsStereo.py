from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
def findDoubleBondsStereo(self):
    """Finds indeces of stereo double bond atoms (E/Z)"""
    db_stereo = {}
    for bond in self.mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            db_stereo[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetStereo()
    return db_stereo