import array
import re
import sys
from chemaxon.descriptors import (CFParameters, ChemicalFingerprint,
from chemaxon.struc import Molecule
from chemaxon.util import MolHandler
def desalt(mol):
    parmol = mol
    smi = mol.toFormat('smiles')
    parcount = 0
    msmi = smi.split('.')
    for smi in msmi:
        mol = MolHandler(smi).getMolecule()
        count = mol.getAtomCount()
        if count > parcount:
            parcount = count
            parmol = mol
    return parmol