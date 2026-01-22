from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_bcc_symmetry_ops(comparator):
    s1 = get_atoms_with_mixed_elements(crystalstructure='bcc')
    s2 = s1.copy()
    sg = spacegroup.Spacegroup(229)
    cell = s2.get_cell().T
    inv_cell = np.linalg.inv(cell)
    operations = sg.get_rotations()
    if not heavy_test:
        operations = operations[::int(np.ceil(len(operations) / 4))]
    for op in operations:
        s1 = get_atoms_with_mixed_elements(crystalstructure='bcc')
        s2 = s1.copy()
        transformed_op = cell.dot(op).dot(inv_cell)
        s2.set_positions(transformed_op.dot(s1.get_positions().T).T)
        assert comparator.compare(s1, s2)