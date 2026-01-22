import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.io.rmc6f as rmc6f
from ase.lattice.compounds import TRI_Fe2O3
def test_rmc6f_write():
    """Test for writing rmc6f input file."""
    tol = 1e-05
    write('output.rmc6f', rmc6f_atoms)
    readback = read('output.rmc6f')
    assert np.allclose(rmc6f_atoms.positions, readback.positions, rtol=tol)
    assert readback == rmc6f_atoms