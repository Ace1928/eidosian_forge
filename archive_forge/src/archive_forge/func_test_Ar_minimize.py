import pytest
from numpy.testing import assert_allclose
from ase.cluster.icosahedron import Icosahedron
from ase.data import atomic_numbers, atomic_masses
from ase.optimize import LBFGS
@pytest.mark.calculator_lite
@pytest.mark.calculator('lammpsrun')
def test_Ar_minimize(factory, ar_nc, params):
    with factory.calc(specorder=['Ar'], **params) as calc:
        ar_nc.calc = calc
        assert_allclose(ar_nc.get_potential_energy(), -0.468147667942117, atol=0.0001, rtol=0.0001)
        assert_allclose(ar_nc.get_forces(), calc.calculate_numerical_forces(ar_nc), atol=0.0001, rtol=0.0001)
        with LBFGS(ar_nc, force_consistent=False) as dyn:
            dyn.run(fmax=1e-06)
        assert_allclose(ar_nc.get_potential_energy(), -0.4791815886953914, atol=0.0001, rtol=0.0001)
        assert_allclose(ar_nc.get_forces(), calc.calculate_numerical_forces(ar_nc), atol=0.0001, rtol=0.0001)