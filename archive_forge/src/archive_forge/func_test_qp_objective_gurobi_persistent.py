import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(gurobi_persistent.available(exception_flag=False), 'needs Gurobi Persistent interface')
def test_qp_objective_gurobi_persistent(self):
    m = self._qp_model()
    gurobi_persistent.set_instance(m)
    results = gurobi_persistent.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])