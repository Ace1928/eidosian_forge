from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_one_atom_out_of_pos(comparator):
    s1 = get_atoms_with_mixed_elements()
    s2 = s1.copy()
    pos = s1.get_positions()
    pos[0, :] += 0.2
    s2.set_positions(pos)
    assert not comparator.compare(s1, s2)