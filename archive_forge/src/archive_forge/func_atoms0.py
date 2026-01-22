import numpy as np
import pytest
from ase.build import molecule
from ase.utils.ff import Morse, Angle, Dihedral, VdW
from ase.calculators.ff import ForceField
from ase.optimize.precon.neighbors import get_neighbours
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.optimize.precon import FF
@pytest.fixture(scope='module')
def atoms0():
    a = molecule('C60')
    a.set_cell(50.0 * np.identity(3))
    return a