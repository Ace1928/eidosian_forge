import pyomo.common.unittest as unittest
from pyomo.opt import TerminationCondition, SolutionStatus, check_available_solvers
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
def test_qcqo(self):
    model = pmo.block()
    model.x = pmo.variable_list()
    for i in range(3):
        model.x.append(pmo.variable(lb=0.0))
    model.cons = pmo.constraint(expr=model.x[0] + model.x[1] + model.x[2] - model.x[0] ** 2 - model.x[1] ** 2 - 0.1 * model.x[2] ** 2 + 0.2 * model.x[0] * model.x[2] >= 1.0)
    model.o = pmo.objective(expr=model.x[0] ** 2 + 0.1 * model.x[1] ** 2 + model.x[2] ** 2 - model.x[0] * model.x[2] - model.x[1], sense=pmo.minimize)
    opt = pmo.SolverFactory('mosek_direct')
    results = opt.solve(model)
    self.assertAlmostEqual(results.problem.upper_bound, -0.49176, 4)
    self.assertAlmostEqual(results.problem.lower_bound, -0.4918, 4)
    del model