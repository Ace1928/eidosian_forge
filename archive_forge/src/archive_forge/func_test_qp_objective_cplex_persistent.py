import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(cplex_persistent.available(exception_flag=False), 'needs Cplex Persistent interface')
def test_qp_objective_cplex_persistent(self):
    m = self._qp_model()
    cplex_persistent.set_instance(m)
    results = cplex_persistent.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])