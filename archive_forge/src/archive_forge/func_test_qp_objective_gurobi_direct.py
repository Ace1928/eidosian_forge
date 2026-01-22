import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(gurobi_direct.available(exception_flag=False), 'needs Gurobi Direct interface')
def test_qp_objective_gurobi_direct(self):
    m = self._qp_model()
    results = gurobi_direct.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])