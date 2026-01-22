from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def test_index_shuffle():
    atoms, constr, bond_def, target_bond, angle_def, target_angle, dihedral_def, target_dihedral = setup_fixinternals()
    constr2 = copy.deepcopy(constr)
    assert all((a == b for a, b in zip(constr.get_indices(), (0, 1, 2, 6, 8))))
    constr.index_shuffle(atoms, range(len(atoms)))
    assert all((a == b for a, b in zip(constr.get_indices(), (0, 1, 2, 6, 8))))
    with pytest.raises(IndexError):
        constr.index_shuffle(atoms, [0])
    constr2.index_shuffle(atoms, [1, 2, 0, 6])
    assert constr2.bonds[0][1] == [0, 1]
    assert constr2.angles[0][1] == [3, 2, 0]
    assert constr2.dihedrals[0][1] == [3, 2, 0, 1]