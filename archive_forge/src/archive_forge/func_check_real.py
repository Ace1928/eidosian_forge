import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode
def check_real(idx, solver, meth, use_jac, with_jac, banded):
    a = real_matrices[idx]
    y0, t_exact, y_exact = real_solutions[idx]
    t, y = _solve_linear_sys(a, y0, tend=t_exact[-1], dt=t_exact[1] - t_exact[0], solver=solver, method=meth, use_jac=use_jac, with_jacobian=with_jac, banded=banded)
    assert_allclose(t, t_exact)
    assert_allclose(y, y_exact)