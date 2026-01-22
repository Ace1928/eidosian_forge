import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def newtype(element, types):
    if len(element) > 1:
        return element
    count = 0
    for type in types:
        if type[0] == element:
            count += 1
    label = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return element + label[count]