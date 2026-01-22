import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.lj import LennardJones
from ase.spacegroup.symmetrize import FixSymmetry, check_symmetry, is_subgroup
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.constraints import UnitCellFilter, ExpCellFilter
@pytest.mark.filterwarnings('ignore:ASE Atoms-like input is deprecated')
def test_fix_symmetry_shuffle_indices():
    atoms = Atoms('AlFeAl6', cell=[6] * 3, positions=[[0, 0, 0], [2.9, 2.9, 2.9], [0, 0, 3], [0, 3, 0], [0, 3, 3], [3, 0, 0], [3, 0, 3], [3, 3, 0]], pbc=True)
    atoms.set_constraint(FixSymmetry(atoms))
    at_permut = atoms[[0, 2, 3, 4, 5, 6, 7, 1]]
    pos0 = atoms.get_positions()

    def perturb(atoms, pos0, at_i, dpos):
        positions = pos0.copy()
        positions[at_i] += dpos
        atoms.set_positions(positions)
        new_p = atoms.get_positions()
        return pos0[at_i] - new_p[at_i]
    dp1 = perturb(atoms, pos0, 1, (0.0, 0.1, -0.1))
    dp2 = perturb(atoms, pos0, 2, (0.0, 0.1, -0.1))
    pos0 = at_permut.get_positions()
    permut_dp1 = perturb(at_permut, pos0, 7, (0.0, 0.1, -0.1))
    permut_dp2 = perturb(at_permut, pos0, 1, (0.0, 0.1, -0.1))
    assert np.max(np.abs(dp1 - permut_dp1)) < 1e-10
    assert np.max(np.abs(dp2 - permut_dp2)) < 1e-10