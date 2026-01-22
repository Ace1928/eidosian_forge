import pytest
import numpy as np
from ase.build import molecule
from ase.md.velocitydistribution import Stationary, ZeroRotation
@pytest.fixture
def stationary_atoms(atoms):
    atoms = atoms.copy()
    Stationary(atoms)
    return atoms