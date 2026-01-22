import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_empty_singleton(self):
    a = Objective()
    a.construct()
    self.assertEqual(a._constructed, True)
    self.assertEqual(len(a), 0)
    try:
        a()
        self.fail('Component is empty')
    except ValueError:
        pass
    try:
        a.expr
        self.fail('Component is empty')
    except ValueError:
        pass
    try:
        a.sense
        self.fail('Component is empty')
    except ValueError:
        pass
    x = Var(initialize=1.0)
    x.construct()
    a.set_value(x + 1)
    self.assertEqual(len(a), 1)
    self.assertEqual(a(), 2)
    self.assertEqual(a.expr(), 2)
    self.assertEqual(a.sense, minimize)