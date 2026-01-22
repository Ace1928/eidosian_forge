import pyomo.common.unittest as unittest
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex
from pyomo.environ import (
from pyomo.opt import WriterFactory, SolverFactory
def test_write_scheduling_model_only_interval_vars(self):
    m = ConcreteModel()
    m.i = IntervalVar(start=(2, 4), end=(5, 19), length=7, optional=False)
    m.tasks = Set(initialize=range(2))
    m.h = IntervalVar(m.tasks, optional=True, length=(4, 5), start=(1, 2))
    cpx_mod, var_map = WriterFactory('docplex_model').write(m)
    exprs = cpx_mod.get_all_expressions()
    self.assertEqual(len(exprs), 3)
    variables = cpx_mod.get_all_variables()
    self.assertEqual(len(variables), 3)
    self.assertIs(variables[0], var_map[m.h[1]])
    self.assertIs(exprs[2][0], var_map[m.h[1]])
    self.assertIs(variables[1], var_map[m.h[0]])
    self.assertIs(exprs[1][0], var_map[m.h[0]])
    for i in [0, 1]:
        self.assertTrue(variables[i].is_optional())
        self.assertEqual(variables[i].get_start(), (1, 2))
        self.assertEqual(variables[i].get_length(), (4, 5))
    self.assertIs(variables[2], var_map[m.i])
    self.assertIs(exprs[0][0], var_map[m.i])
    self.assertTrue(variables[2].is_present())
    self.assertEqual(variables[2].get_start(), (2, 4))
    self.assertEqual(variables[2].get_end(), (5, 19))
    self.assertEqual(variables[2].get_length(), (7, 7))