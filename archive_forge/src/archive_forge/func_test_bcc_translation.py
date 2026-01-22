from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_bcc_translation(comparator):
    s1 = get_atoms_with_mixed_elements(crystalstructure='bcc')
    s2 = s1.copy()
    s2.set_positions(s2.get_positions() + np.array([6.0, -2.0, 1.0]))
    assert comparator.compare(s1, s2)