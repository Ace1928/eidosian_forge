import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms, UnitCellFilter
from ase.calculators.emt import EMT
from ase.optimize.precon import make_precon, Precon
from ase.neighborlist import neighbor_list
from ase.utils.ff import Bond
@pytest.mark.parametrize('precon', precons)
def test_assembly_ref_atoms(precon, ref_atoms):
    check_assembly(precon, ref_atoms)