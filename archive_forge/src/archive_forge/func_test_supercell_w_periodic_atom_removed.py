from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_supercell_w_periodic_atom_removed(comparator):
    s1 = Atoms(['H', 'H', 'He', 'He', 'He'], positions=[[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.4, 0.3, 0.2], [0.3, 0.6, 0.3], [0.8, 0.5, 0.6]], cell=[1, 1, 1], pbc=True)
    s1 *= (2, 1, 1)
    a0 = s1.copy()
    del a0[0]
    a5 = s1.copy()
    del a5[5]
    assert comparator.compare(a0, a5)
    assert comparator.compare(a5, a0) == comparator.compare(a0, a5)