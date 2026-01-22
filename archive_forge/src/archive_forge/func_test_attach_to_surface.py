import pytest
import numpy as np
from ase.parallel import world
from ase.build import molecule, fcc111
from ase.build.attach import (attach, attach_randomly,
def test_attach_to_surface():
    """Attach a molecule to a surafce at a given distance"""
    slab = fcc111('Al', size=(3, 2, 2), vacuum=10.0)
    mol = molecule('CH4')
    distance = 3.0
    struct = attach(slab, mol, distance, (0, 0, 1))
    dmin = np.linalg.norm(struct[6].position - struct[15].position)
    assert dmin == pytest.approx(distance, 1e-08)