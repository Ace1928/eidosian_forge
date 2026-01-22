import copy
import numpy as np
import cvxpy as cp
from cvxpy.constraints import Equality
def test_problem():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize((x + y) ** 2)
    constraints = [x + y == 1]
    prob = cp.Problem(obj, constraints)
    prob_copy = copy.copy(prob)
    prob_deepcopy = copy.deepcopy(prob)
    assert id(prob) != id(prob_copy)
    assert id(prob) != id(prob_deepcopy)
    assert id(prob_copy) != id(prob_deepcopy)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    prob_copy.solve()
    assert prob_copy.status == cp.OPTIMAL
    prob_deepcopy.solve()
    assert prob_deepcopy.status == cp.OPTIMAL