import pytest
import numpy as np
from ase.build import molecule
from ase.md.velocitydistribution import Stationary, ZeroRotation
def test_stationary_propagate(atoms, stationary_atoms):
    prop_atoms = propagate(atoms)
    stationary_prop_atoms = propagate(stationary_atoms)
    com = atoms.get_center_of_mass()
    assert norm(prop_atoms.get_center_of_mass() - com) > 0.0001
    assert norm(stationary_prop_atoms.get_center_of_mass() - com) < 1e-13