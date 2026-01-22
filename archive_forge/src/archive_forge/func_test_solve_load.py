import tempfile
import os
import pickle
import random
import collections
import itertools
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.symbol_map import SymbolMap
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import (
from pyomo.common.collections import ComponentMap
from pyomo.core.kernel.suffix import suffix
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.parameter import parameter, parameter_dict, parameter_list
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.objective import objective, objective_dict, objective_list
from pyomo.core.kernel.variable import IVariable, variable, variable_dict, variable_list
from pyomo.core.kernel.block import IBlock, block, block_dict, block_tuple, block_list
from pyomo.core.kernel.sos import sos
from pyomo.opt.results import Solution
def test_solve_load(self):
    b = block()
    b.v = variable()
    b.y = variable(value=1.0, fixed=True)
    b.p = parameter(value=2.0)
    b.e = expression(b.v * b.p + 1)
    b.eu = data_expression(b.p ** 2 + 10)
    b.el = data_expression(-b.p ** 2 - 10)
    b.o = objective(b.v + b.y)
    b.c1 = constraint((b.el + 1, b.e + 2, b.eu + 2))
    b.c2 = constraint(lb=b.el, body=b.v)
    b.c3 = constraint(body=b.v, ub=b.eu)
    b.dual = suffix(direction=suffix.IMPORT)
    import pyomo.environ
    from pyomo.opt.base.solvers import UnknownSolver
    from pyomo.opt import SolverFactory
    from pyomo.opt import SolverStatus, TerminationCondition
    opt = SolverFactory('glpk')
    if isinstance(opt, UnknownSolver) or not opt.available():
        raise unittest.SkipTest('glpk solver not available')
    status = opt.solve(b)
    self.assertEqual(status.solver.status, SolverStatus.ok)
    self.assertEqual(status.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(b.o(), -7, places=5)
    self.assertAlmostEqual(b.v(), -8, places=5)
    self.assertAlmostEqual(b.y(), 1.0, places=5)
    opt = SolverFactory('glpk')
    if isinstance(opt, UnknownSolver) or not opt.available():
        raise unittest.SkipTest('glpk solver not available')
    status = opt.solve(b, symbolic_solver_labels=True)
    self.assertEqual(status.solver.status, SolverStatus.ok)
    self.assertEqual(status.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(b.o(), -7, places=5)
    self.assertAlmostEqual(b.v(), -8, places=5)
    self.assertAlmostEqual(b.y(), 1.0, places=5)
    opt = SolverFactory('ipopt')
    if isinstance(opt, UnknownSolver) or not opt.available():
        raise unittest.SkipTest('ipopt solver not available')
    status = opt.solve(b)
    self.assertEqual(status.solver.status, SolverStatus.ok)
    self.assertEqual(status.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(b.o(), -7, places=5)
    self.assertAlmostEqual(b.v(), -8, places=5)
    self.assertAlmostEqual(b.y(), 1.0, places=5)
    opt = SolverFactory('ipopt')
    if isinstance(opt, UnknownSolver):
        raise unittest.SkipTest('ipopt solver not available')
    status = opt.solve(b, symbolic_solver_labels=True)
    self.assertEqual(status.solver.status, SolverStatus.ok)
    self.assertEqual(status.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(b.o(), -7, places=5)
    self.assertAlmostEqual(b.v(), -8, places=5)
    self.assertAlmostEqual(b.y(), 1.0, places=5)