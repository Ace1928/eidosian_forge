from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def setup_combos():
    atoms = setup_atoms()
    bondcombo_def = [[2, 1, 1.0], [2, 3, -1.0]]
    target_bondcombo = get_bondcombo(atoms, bondcombo_def)
    anglecombo_def = [[7, 0, 8, 1.0], [7, 0, 6, 1]]
    target_anglecombo = get_anglecombo(atoms, anglecombo_def)
    dihedralcombo_def = [[3, 2, 1, 4, 1.0], [2, 1, 0, 7, 1.0]]
    target_dihedralcombo = get_dihedralcombo(atoms, dihedralcombo_def)
    constr = FixInternals(bondcombos=[(target_bondcombo, bondcombo_def)], anglecombos=[(target_anglecombo, anglecombo_def)], dihedralcombos=[(target_dihedralcombo, dihedralcombo_def)], epsilon=1e-10)
    print(constr)
    return (atoms, constr, bondcombo_def, target_bondcombo, anglecombo_def, target_anglecombo, dihedralcombo_def, target_dihedralcombo)