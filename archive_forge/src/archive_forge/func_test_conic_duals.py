import pyomo.common.unittest as unittest
from pyomo.opt import TerminationCondition, SolutionStatus, check_available_solvers
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
def test_conic_duals(self):
    check = [-1.94296808, -0.303030303, -1.91919191]
    with pmo.SolverFactory('mosek_direct') as solver:
        model = self._test_model()
        results = solver.solve(model)
        model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)
        solver.load_duals()
        for i in range(3):
            self.assertAlmostEqual(model.dual[model.quad.q][i], check[i], 5)
    with pmo.SolverFactory('mosek_direct') as solver:
        model = self._test_model()
        results = solver.solve(model)
        model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)
        solver.load_duals([model.quad.q])
        for i in range(3):
            self.assertAlmostEqual(model.dual[model.quad.q][i], check[i], 5)
    with pmo.SolverFactory('mosek_direct') as solver:
        model = self._test_model()
        model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)
        results = solver.solve(model, save_results=True)
        for i in range(3):
            self.assertAlmostEqual(results.Solution.constraint['x11']['Dual'][i], check[i], 5)