import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_valid_declaration(self):
    model = ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1))
    self.assertEqual(len(model.t), 2)
    self.assertIn(0, model.t)
    self.assertIn(1, model.t)
    model = ConcreteModel()
    model.t = ContinuousSet(initialize=[1, 2, 3])
    self.assertEqual(len(model.t), 3)
    self.assertEqual(model.t.first(), 1)
    self.assertEqual(model.t.last(), 3)
    model = ConcreteModel()
    model.t = ContinuousSet(bounds=(1, 3), initialize=[1, 2, 3])
    self.assertEqual(len(model.t), 3)
    self.assertEqual(model.t.first(), 1)
    self.assertEqual(model.t.last(), 3)
    model = ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 4), initialize=[1, 2, 3])
    self.assertEqual(len(model.t), 5)
    self.assertEqual(model.t.first(), 0)
    self.assertEqual(model.t.last(), 4)
    model = ConcreteModel()
    with self.assertRaisesRegex(ValueError, 'value is not in the domain \\[0..4\\]'):
        model.t = ContinuousSet(bounds=(0, 4), initialize=[1, 2, 3, 5])
    model = ConcreteModel()
    with self.assertRaisesRegex(ValueError, 'value is not in the domain \\[2..6\\]'):
        model.t = ContinuousSet(bounds=(2, 6), initialize=[1, 2, 3, 5])
    model = ConcreteModel()
    with self.assertRaisesRegex(ValueError, 'value is not in the domain \\[2..4\\]'):
        model.t = ContinuousSet(bounds=(2, 4), initialize=[1, 3, 5])