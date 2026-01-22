import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(cplex_lp.available(exception_flag=False), 'needs Cplex LP interface')
def test_qp_objective_cplex_lp(self):
    m = self._qp_model()
    results = cplex_lp.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])