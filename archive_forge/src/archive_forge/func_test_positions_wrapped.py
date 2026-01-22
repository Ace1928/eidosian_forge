import numpy as np
from ase import Atoms
def test_positions_wrapped(atoms=atoms):
    assert np.allclose(positions_wrapped, atoms.get_positions(wrap=True))