import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms, UnitCellFilter
from ase.calculators.emt import EMT
from ase.optimize.precon import make_precon, Precon
from ase.neighborlist import neighbor_list
from ase.utils.ff import Bond
@pytest.fixture
def var_cell(atoms):
    atoms, bonds = atoms
    return (UnitCellFilter(atoms), bonds)