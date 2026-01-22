import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
@unittest.skipIf(not xpress_available, 'xpress is not available')
def test_nonconvexqp_infeasible(self):
    """Test non-convex QP which xpress_direct should prove infeasible."""
    m = pe.ConcreteModel()
    m.x1 = pe.Var()
    m.x2 = pe.Var()
    m.x3 = pe.Var()
    m.obj = pe.Objective(rule=lambda m: 2 * m.x1 + m.x2 + m.x3, sense=pe.minimize)
    m.equ1a = pe.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == 1)
    m.equ1b = pe.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == -1)
    m.cone = pe.Constraint(rule=lambda m: m.x2 * m.x2 + m.x3 * m.x3 <= m.x1 * m.x1)
    m.equ2 = pe.Constraint(rule=lambda m: m.x1 >= 0)
    opt = pe.SolverFactory('xpress_direct')
    opt.options['XSLP_SOLVER'] = 0
    results = opt.solve(m)
    self.assertEqual(results.solver.status, SolverStatus.ok)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)