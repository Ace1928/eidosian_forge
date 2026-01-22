from functools import partial
import pytest
from ase.calculators.emt import EMT
from ase.optimize import (MDMin, FIRE, LBFGS, LBFGSLineSearch, BFGSLineSearch,
from ase.optimize.sciopt import SciPyFminCG, SciPyFminBFGS
from ase.optimize.precon import PreconFIRE, PreconLBFGS, PreconODE12r
from ase.cluster import Icosahedron
from ase.build import bulk
@pytest.fixture(scope='module')
def ref_atoms():
    atoms = bulk('Au')
    atoms.calc = EMT()
    atoms.get_potential_energy()
    return atoms