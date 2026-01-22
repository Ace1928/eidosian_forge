from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_order_of_candidates(comparator):
    s1 = bulk('Al', crystalstructure='fcc', a=3.2)
    s1 = s1 * (2, 2, 2)
    s2 = s1.copy()
    s1.positions[0, :] += 0.2
    assert comparator.compare(s2, s1) == comparator.compare(s1, s2)