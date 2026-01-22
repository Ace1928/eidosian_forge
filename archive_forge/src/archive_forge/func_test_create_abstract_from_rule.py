import json
import os
from os.path import abspath, dirname, join
import pickle
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.tempfiles import TempfileManager
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.opt import check_available_solvers
from pyomo.opt.parallel.local import SolverManager_Serial
def test_create_abstract_from_rule(self):

    def make_invalid(m):
        m.I = RangeSet(3)
        m.x = Var(m.I)
        m.c = Constraint(expr=sum((m.x[i] for i in m.I)) >= 0)

    def make(m):
        m.I = RangeSet(3)
        m.x = Var(m.I)

        def c(b):
            return sum((m.x[i] for i in m.I)) >= 0
        m.c = Constraint(rule=c)
    with self.assertRaisesRegex(ValueError, 'x\\[1\\]: The component has not been constructed.'):
        model = AbstractModel(rule=make_invalid)
        instance = model.create_instance()
    model = AbstractModel(rule=make)
    instance = model.create_instance()
    self.assertEqual([x.local_name for x in model.component_objects()], [])
    self.assertEqual([x.local_name for x in instance.component_objects()], ['I', 'x', 'c'])
    self.assertEqual(len(list(EXPR.identify_variables(instance.c.body))), 3)
    model = AbstractModel(rule=make)
    model.y = Var()
    instance = model.create_instance()
    self.assertEqual([x.local_name for x in instance.component_objects()], ['y', 'I', 'x', 'c'])
    self.assertEqual(len(list(EXPR.identify_variables(instance.c.body))), 3)