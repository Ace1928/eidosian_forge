import pytest
from ase.lattice.cubic import FaceCenteredCubic
@pytest.fixture
def lattice_params():
    lattice_params = {}
    lattice_params['size'] = (2, 2, 2)
    lattice_params['latticeconstant'] = 3.52
    lattice_params['symbol'] = 'Ni'
    lattice_params['pbc'] = True
    return lattice_params