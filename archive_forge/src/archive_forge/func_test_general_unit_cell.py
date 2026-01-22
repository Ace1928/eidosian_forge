from math import pi, sqrt, cos
import pytest
import numpy as np
from ase import Atoms
from ase import data
from ase.lattice.cubic import FaceCenteredCubic
def test_general_unit_cell(atoms_guc):
    atoms = atoms_guc
    assert len(atoms) == 5 * 5 * 5 * 2
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 4)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(2.5 * a0 - c[2, 2]) < 1e-10