import pytest
import numpy as np
from ase.io import write
@calc('vasp')
def test_vasp_co(factory, atoms_co, calc_settings):
    """
    Run some VASP tests to ensure that the VASP calculator works. This
    is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
    environment variables

    """

    def array_almost_equal(a1, a2, tol=np.finfo(type(1.0)).eps):
        """Replacement for old numpy.testing.utils.array_almost_equal."""
        return (np.abs(a1 - a2) < tol).all()
    co = atoms_co
    calc = factory.calc(**calc_settings)
    co.calc = calc
    en = co.get_potential_energy()
    write('vasp_co.traj', co)
    calc2 = factory.calc(restart=True)
    co2 = calc2.get_atoms()
    assert array_almost_equal(co.positions, co2.positions, 1e-14)
    assert en - co2.get_potential_energy() == pytest.approx(0)
    assert array_almost_equal(calc.get_stress(co), calc2.get_stress(co2))
    assert array_almost_equal(calc.get_forces(co), calc2.get_forces(co2))
    assert array_almost_equal(calc.get_eigenvalues(), calc2.get_eigenvalues())
    assert calc.get_number_of_bands() == calc2.get_number_of_bands()
    assert calc.get_xc_functional() == calc2.get_xc_functional()
    calc.clean()