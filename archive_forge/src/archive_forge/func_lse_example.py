import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
@pytest.fixture
def lse_example():
    ref_x = cp.Variable(3)
    ref_s = cp.Variable()
    ref_z = cp.Variable(3)
    ref_t = cp.Variable()
    ref_constraints = [ref_s >= cp.sum(ref_z), [1, 2, 3] <= ref_x, 1 <= ref_s, ref_s <= 2]
    ref_constraints += [ExpCone(ref_x[i] - ref_t, ref_s, ref_z[i]) for i in range(3)]
    ref_prob = cp.Problem(cp.Minimize(ref_t), ref_constraints)
    ref_prob.solve(solver=cp.ECOS)
    return (ref_prob.value, ref_x.value, ref_s.value)