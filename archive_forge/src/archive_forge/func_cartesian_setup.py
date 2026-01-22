import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
@pytest.fixture
def cartesian_setup():
    positions = [[-0.464, 0.177, 0.0], [-0.464, 1.137, 0.0], [0.441, -0.143, 0.0]]
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    masses = [15.999, 0.1134289259, 2]
    atoms = Atoms('OH2', cell=cell, positions=positions, masses=masses, pbc=True)
    params = {'chk': 'example.chk', 'nprocshared': '16', 'output_type': 'n', 'method': 'b3lyp', 'basis': "6-31g(d',p')", 'opt': 'tight, maxcyc=100', 'integral': 'ultrafine', 'charge': 0, 'mult': 1, 'isolist': np.array([None, 0.1134289259, 2])}
    return (atoms, params)