from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_symmetrical_one_element_out(comparator):
    s1 = get_atoms_with_mixed_elements()
    s1.set_chemical_symbols(['Zn', 'Zn', 'Al', 'Zn', 'Zn', 'Al', 'Zn', 'Zn'])
    s2 = s1.copy()
    s2.positions[0, :] += 0.2
    assert not comparator.compare(s1, s2)
    assert not comparator.compare(s2, s1)