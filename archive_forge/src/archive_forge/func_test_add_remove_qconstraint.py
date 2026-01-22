import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
@unittest.skipIf(not xpress_available, 'xpress is not available')
def test_add_remove_qconstraint(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.z = pe.Var()
    m.obj = pe.Objective(expr=m.z)
    m.c1 = pe.Constraint(expr=m.z >= m.x ** 2 + m.y ** 2)
    opt = pe.SolverFactory('xpress_persistent')
    opt.set_instance(m)
    self.assertEqual(opt.get_xpress_attribute('rows'), 1)
    opt.remove_constraint(m.c1)
    self.assertEqual(opt.get_xpress_attribute('rows'), 0)
    opt.add_constraint(m.c1)
    self.assertEqual(opt.get_xpress_attribute('rows'), 1)