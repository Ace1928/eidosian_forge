import pytest
import numpy as np
from ase.collections import g2
from ase.build import bulk, graphene_nanoribbon
@calc('octopus', Spacing='0.25 * angstrom')
@pytest.mark.xfail
def test_h2o(factory):
    calc = calculate(factory, g2['H2O'], OutputFormat='xcrysden', SCFCalculateDipole=True)
    dipole = calc.get_dipole_moment()
    E = calc.get_potential_energy()
    print('dipole', dipole)
    print('energy', E)
    assert pytest.approx(dipole, abs=0.02) == [0, 0, -0.37]
    dipole_err = np.abs(dipole - [0.0, 0.0, -0.37]).max()
    assert dipole_err < 0.02, dipole_err