import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
@unittest.skipIf(not xpress_available, 'xpress is not available')
def test_add_column_exceptions(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.c = pe.Constraint(expr=(0, m.x, 1))
    m.ci = pe.Constraint([1, 2], rule=lambda m, i: (0, m.x, i + 1))
    m.cd = pe.Constraint(expr=(0, -m.x, 1))
    m.cd.deactivate()
    m.obj = pe.Objective(expr=-m.x)
    opt = pe.SolverFactory('xpress_persistent')
    self.assertRaises(RuntimeError, opt.add_column, m, m.x, 0, [m.c], [1])
    opt.set_instance(m)
    m2 = pe.ConcreteModel()
    m2.y = pe.Var()
    m2.c = pe.Constraint(expr=(0, m.x, 1))
    self.assertRaises(RuntimeError, opt.add_column, m2, m2.y, 0, [], [])
    self.assertRaises(RuntimeError, opt.add_column, m, m2.y, 0, [], [])
    z = pe.Var()
    self.assertRaises(RuntimeError, opt.add_column, m, z, -2, [m.c, z], [1])
    m.y = pe.Var()
    self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1, 2])
    self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c, z], [1])
    self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.ci], [1])
    self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.x], [1])
    self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m2.c], [1])
    self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m.cd], [1])
    opt.add_var(m.y)
    self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1])