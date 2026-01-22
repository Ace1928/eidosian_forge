from pathlib import Path
import numpy as np
import pytest
import ase.io
from ase.io import extxyz
from ase.atoms import Atoms
from ase.build import bulk
from ase.io.extxyz import escape
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixCartesian
from ase.stress import full_3x3_to_voigt_6_stress
from ase.build import molecule
@pytest.mark.filterwarnings('ignore:write_xyz')
def test_stress():
    water1 = molecule('H2O')
    water2 = molecule('H2O')
    water2.positions[:, 0] += 5.0
    atoms = water1 + water2
    atoms.cell = [10, 10, 10]
    atoms.pbc = True
    atoms.new_array('stress', np.arange(6, dtype=float))
    atoms.calc = EMT()
    a_stress = atoms.get_stress()
    atoms.write('tmp.xyz')
    b = ase.io.read('tmp.xyz')
    assert abs(b.get_stress() - a_stress).max() < 1e-06
    assert abs(b.arrays['stress'] - np.arange(6, dtype=float)).max() < 1e-06
    b_stress = b.info['stress']
    assert abs(full_3x3_to_voigt_6_stress(b_stress) - a_stress).max() < 1e-06