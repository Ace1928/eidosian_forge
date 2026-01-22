import numpy.random as random
import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import (NeighborList, PrimitiveNeighborList,
from ase.build import bulk
def test_small_cell_and_large_cutoff():
    cutoff = 50
    atoms = bulk('Cu', 'fcc', a=3.6)
    atoms *= (2, 2, 2)
    atoms.set_pbc(False)
    radii = cutoff * np.ones(len(atoms.get_atomic_numbers()))
    neighborhood_new = NeighborList(radii, skin=0.0, self_interaction=False, bothways=True, primitive=NewPrimitiveNeighborList)
    neighborhood_old = NeighborList(radii, skin=0.0, self_interaction=False, bothways=True, primitive=PrimitiveNeighborList)
    neighborhood_new.update(atoms)
    neighborhood_old.update(atoms)
    n0, d0 = neighborhood_new.get_neighbors(0)
    n1, d1 = neighborhood_old.get_neighbors(0)
    assert np.all(n0 == n1)
    assert np.all(d0 == d1)