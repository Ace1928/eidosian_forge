import pytest
from ase.atoms import Atoms
@pytest.fixture
def atoms_2():
    return Atoms('CaInI', positions=[[0, 0, 1], [0, 0, 2], [0, 0, 3]], cell=[5, 5, 5])