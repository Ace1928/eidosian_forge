import pytest
import numpy as np
from ase.build import bulk
from ase.constraints import UnitCellFilter
from ase.optimize import MDMin
@pytest.mark.calculator_lite
def test_cp2k_stress(cp2k_factory):
    """Adopted from ase/test/stress.py"""
    inp = '&FORCE_EVAL\n                  &MM\n                    &FORCEFIELD\n                      &SPLINE\n                        EMAX_ACCURACY 500.0\n                        EMAX_SPLINE    1000.0\n                        EPS_SPLINE 1.0E-9\n                      &END\n                      &NONBONDED\n                        &LENNARD-JONES\n                          atoms Ar Ar\n                          EPSILON [eV] 1.0\n                          SIGMA [angstrom] 1.0\n                          RCUT [angstrom] 10.0\n                        &END LENNARD-JONES\n                      &END NONBONDED\n                      &CHARGE\n                        ATOM Ar\n                        CHARGE 0.0\n                      &END CHARGE\n                    &END FORCEFIELD\n                    &POISSON\n                      &EWALD\n                        EWALD_TYPE none\n                      &END EWALD\n                    &END POISSON\n                  &END MM\n                &END FORCE_EVAL'
    calc = cp2k_factory.calc(label='test_stress', inp=inp, force_eval_method='Fist')
    vol0 = 4 * 0.91615977036
    a0 = vol0 ** (1 / 3)
    a = bulk('Ar', 'fcc', a=a0)
    cell0 = a.get_cell()
    a.calc = calc
    a.set_cell(np.dot(a.cell, [[1.02, 0, 0.03], [0, 0.99, -0.02], [0.1, -0.01, 1.03]]), scale_atoms=True)
    a *= (1, 2, 3)
    cell0 *= np.array([1, 2, 3])[:, np.newaxis]
    a.rattle()
    s_analytical = a.get_stress()
    s_numerical = a.calc.calculate_numerical_stress(a, 1e-05)
    s_p_err = 100 * (s_numerical - s_analytical) / s_numerical
    print('Analytical stress:\n', s_analytical)
    print('Numerical stress:\n', s_numerical)
    print('Percent error in stress:\n', s_p_err)
    assert np.all(abs(s_p_err) < 1e-05)
    opt = MDMin(UnitCellFilter(a), dt=0.01)
    opt.run(fmax=0.001)
    g_minimized = np.dot(a.cell, a.cell.T)
    g_theory = np.dot(cell0, cell0.T)
    g_p_err = 100 * (g_minimized - g_theory) / g_theory
    print('Minimized Niggli tensor:\n', g_minimized)
    print('Theoretical Niggli tensor:\n', g_theory)
    print('Percent error in Niggli tensor:\n', g_p_err)
    assert np.all(abs(g_p_err) < 1)
    print('passed test "stress"')