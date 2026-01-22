import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def read_list(key_string, length, debug=False):
    if key != key_string:
        return ([], key)
    lst = []
    while len(lines):
        w = lines.pop(0).split()
        if len(w) > length:
            lst.append([int(w[1 + c]) - 1 for c in range(length)])
        else:
            return (lst, next_key())
    return (lst, None)