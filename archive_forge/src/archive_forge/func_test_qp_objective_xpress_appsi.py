import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
@unittest.skipUnless(xpress_appsi.available(exception_flag=False), 'needs Xpress APPSI interface')
def test_qp_objective_xpress_appsi(self):
    m = self._qp_model()
    xpress_appsi.set_instance(m)
    results = xpress_appsi.solve(m)
    self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])