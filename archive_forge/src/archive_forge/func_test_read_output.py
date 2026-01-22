import numpy as np
import pytest
from ase import Atoms
from ase.io import qbox
from ase.io import formats
def test_read_output(qboxfile):
    """Test reading the output file"""
    atoms = qbox.read_qbox(qboxfile)
    assert isinstance(atoms, Atoms)
    assert np.allclose(atoms.cell, np.diag([16, 16, 16]))
    assert len(atoms) == 4
    assert np.allclose(atoms[0].position, [3.70001108, -0.0, -3e-08], atol=1e-07)
    assert np.allclose(atoms.get_velocities()[2], [-8.9e-07, -0.0, -0.0], atol=1e-09)
    assert np.allclose(atoms.get_forces()[3], [-2.6e-07, -0.01699708, 7.46e-06], atol=1e-07)
    assert np.isclose(-15.37294664, atoms.get_potential_energy())
    assert np.allclose(atoms.get_stress(), [-0.40353661, -1.11698386, -1.39096418, 1.786e-05, -2.405e-05, -1.4e-07])
    atoms = qbox.read_qbox(qboxfile, slice(None))
    assert isinstance(atoms, list)
    assert len(atoms) == 5
    assert len(atoms[1]) == 4
    assert np.allclose(atoms[1][0].position, [3.70001108, -0.0, -3e-08], atol=1e-07)
    assert np.allclose(atoms[1].get_forces()[3], [-2.9e-07, -0.01705361, 7.63e-06], atol=1e-07)