from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_one_vs_many():
    s1 = Atoms('H3', positions=[[0.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]], cell=[2, 2, 2], pbc=True)
    u = (s1.get_volume() / len(s1)) ** (1 / 3)
    comp = SymmetryEquivalenceCheck(stol=0.095 / u, scale_volume=True)
    s2 = s1.copy()
    assert comp.compare(s1, s2)
    s2_list = []
    s3 = Atoms('H3', positions=[[0.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]], cell=[3, 3, 3], pbc=True)
    s2_list.append(s3)
    for d in np.linspace(0.1, 1.0, 5):
        s2 = s1.copy()
        s2.positions[0] += [d, 0, 0]
        s2_list.append(s2)
    assert not comp.compare(s1, s2_list[:-1])
    assert comp.compare(s1, s2_list)