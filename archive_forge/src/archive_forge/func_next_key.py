import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def next_key():
    while len(lines):
        line = lines.pop(0).strip()
        if len(line) > 0:
            lines.pop(0)
            return line
    return None