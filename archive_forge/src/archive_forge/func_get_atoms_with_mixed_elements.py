from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def get_atoms_with_mixed_elements(crystalstructure='fcc'):
    atoms = bulk('Al', crystalstructure=crystalstructure, a=3.2)
    atoms = atoms * (2, 2, 2)
    symbs = ['Al', 'Cu', 'Zn']
    symbols = [symbs[randint(0, len(symbs) - 1)] for _ in range(len(atoms))]
    for i in range(len(atoms)):
        atoms[i].symbol = symbols[i]
    return atoms