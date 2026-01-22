import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
def test_activate_deactivate_indexed(self):
    model = ConcreteModel()
    model.s = Set(initialize=[1])
    model.y = Var(model.s)
    model.x = Var(model.s, bounds=(-1, 1))
    args = ([1], model.y, model.x)
    keywords = {'pw_pts': {1: [-1, 0, 1]}, 'pw_constr_type': 'EQ', 'f_rule': lambda model, i, x: x ** 2}
    model.c = Piecewise(*args, **keywords)
    self.assertTrue(len(model.c[1].component_map(Constraint)) > 0)
    self.assertTrue(len(model.c[1].component_map(Constraint, active=True)) > 0)
    self.assertEqual(model.c.active, True)
    self.assertEqual(model.c[1].active, True)
    model.c[1].deactivate()
    self.assertTrue(len(model.c[1].component_map(Constraint)) > 0)
    self.assertTrue(len(model.c[1].component_map(Constraint, active=True)) > 0)
    self.assertEqual(model.c.active, True)
    self.assertEqual(model.c[1].active, False)
    model.c[1].activate()
    self.assertTrue(len(model.c[1].component_map(Constraint)) > 0)
    self.assertTrue(len(model.c[1].component_map(Constraint, active=True)) > 0)
    self.assertEqual(model.c.active, True)
    self.assertEqual(model.c[1].active, True)
    model.c.deactivate()
    self.assertTrue(len(model.c[1].component_map(Constraint)) > 0)
    self.assertTrue(len(model.c[1].component_map(Constraint, active=True)) > 0)
    self.assertEqual(model.c.active, False)
    self.assertEqual(model.c[1].active, False)