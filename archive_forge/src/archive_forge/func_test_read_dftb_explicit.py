import numpy as np
from io import StringIO
from ase.atoms import Atoms
from ase.units import AUT, Bohr, second
from ase.io.dftb import (read_dftb, read_dftb_lattice,
def test_read_dftb_explicit():
    x = 1.356773
    positions = [[0.0, 0.0, 0.0], [x, x, x]]
    cell = [[2 * x, 2 * x, 0.0], [0.0, 2 * x, 2 * x], [2 * x, 0.0, 2 * x]]
    atoms = Atoms('GaAs', cell=cell, positions=positions, pbc=True)
    atoms_new = read_dftb(fd_explicit)
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.positions, atoms.positions)
    assert np.all(atoms_new.pbc == atoms.pbc)
    assert np.allclose(atoms_new.cell, atoms.cell)