import argparse
import logging
import sys
from rdkit import Chem
from rdkit.Chem.MolStandardize import Standardizer, Validator
def standardize_main(args):
    mol = _read_mol(args)
    s = Standardizer()
    mol = s.standardize(mol)
    _write_mol(mol, args)