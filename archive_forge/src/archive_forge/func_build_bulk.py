import sys
import numpy as np
from ase.db import connect
from ase.build import bulk
from ase.io import read, write
from ase.visualize import view
from ase.build import molecule
from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import ground_state_magnetic_moments
from ase.data import atomic_numbers, covalent_radii
def build_bulk(args):
    L = args.lattice_constant.replace(',', ' ').split()
    d = dict([(key, float(x)) for key, x in zip('ac', L)])
    atoms = bulk(args.name, crystalstructure=args.crystal_structure, a=d.get('a'), c=d.get('c'), orthorhombic=args.orthorhombic, cubic=args.cubic)
    M, X = {'Fe': (2.3, 'bcc'), 'Co': (1.2, 'hcp'), 'Ni': (0.6, 'fcc')}.get(args.name, (None, None))
    if M is not None and args.crystal_structure == X:
        atoms.set_initial_magnetic_moments([M] * len(atoms))
    return atoms