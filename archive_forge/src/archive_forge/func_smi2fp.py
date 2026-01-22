import argparse
import os
import pickle
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
def smi2fp(molid, smiles):
    mol = Chem.MolFromSmiles(smiles)
    onbits = AllChem.GetMorganFingerprintAsBitVect(mol, 2).GetOnBits()
    row = molid
    for bit in onbits:
        row += '\tFP_{}\t1.0'.format(bit)
    row += '\n'
    return row