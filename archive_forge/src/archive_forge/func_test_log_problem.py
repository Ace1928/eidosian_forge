import math
import numpy as np
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.tests.base_test import BaseTest
def test_log_problem(self) -> None:
    x = cp.Variable(2)
    var_dict = {x.id: x}
    obj = cp.Maximize(cp.sum(cp.log(x)))
    constr = [x <= [1, math.e]]
    problem = cp.Problem(obj, constr)
    data, _, _ = problem.get_problem_data(solver=cp.SCS)
    param_cone_prog = data[cp.settings.PARAM_PROB]
    solver = SCS()
    raw_solution = solver.solve_via_data(data, warm_start=False, verbose=False, solver_opts={})['x']
    sltn_dict = param_cone_prog.split_solution(raw_solution, active_vars=var_dict)
    adjoint = param_cone_prog.split_adjoint(sltn_dict)
    self.assertEqual(adjoint.shape, raw_solution.shape)
    for value in sltn_dict[x.id]:
        self.assertTrue(any(value == adjoint))
    problem.solve(cp.SCS)
    self.assertItemsAlmostEqual(x.value, sltn_dict[x.id])
    obj = cp.Minimize(sum(x))
    constr = [cp.log(x) >= 0, x <= [1, 1]]
    problem = cp.Problem(obj, constr)
    data, _, _ = problem.get_problem_data(solver=cp.SCS)
    param_cone_prog = data[cp.settings.PARAM_PROB]
    solver = SCS()
    raw_solution = solver.solve_via_data(data, warm_start=False, verbose=False, solver_opts={})['x']
    sltn_dict = param_cone_prog.split_solution(raw_solution, active_vars=var_dict)
    adjoint = param_cone_prog.split_adjoint(sltn_dict)
    self.assertEqual(adjoint.shape, raw_solution.shape)
    for value in sltn_dict[x.id]:
        self.assertTrue(any(value == adjoint))
    self.assertItemsAlmostEqual(param_cone_prog.split_adjoint(sltn_dict), raw_solution)
    problem.solve(solver=cp.SCS)
    self.assertItemsAlmostEqual(x.value, sltn_dict[x.id])