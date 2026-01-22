import os
import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp
@pytest.fixture
def atoms_co():
    """Simple atoms object for testing with a single CO molecule"""
    d = 1.14
    atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)], pbc=True)
    atoms.center(vacuum=5)
    return atoms