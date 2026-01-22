import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.nonconvex1 import Nonconvex1
from pyomo.contrib.mindtpy.tests.nonconvex2 import Nonconvex2
from pyomo.contrib.mindtpy.tests.nonconvex3 import Nonconvex3
from pyomo.contrib.mindtpy.tests.nonconvex4 import Nonconvex4
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
from pyomo.contrib.mcpp import pyomo_mcpp
@unittest.skipUnless(SolverFactory('gurobi_persistent').available(exception_flag=False) and SolverFactory('gurobi_direct').available(), 'gurobi_persistent and gurobi_direct solver is not available')
def test_GOA_Gurobi(self):
    """Test the global outer approximation decomposition algorithm."""
    with SolverFactory('mindtpy') as opt:
        for model in model_list:
            model = model.clone()
            results = opt.solve(model, strategy='GOA', mip_solver='gurobi_persistent', nlp_solver=required_solvers[0], single_tree=True)
            self.assertIn(results.solver.termination_condition, [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=2)
            self.check_optimal_solution(model)