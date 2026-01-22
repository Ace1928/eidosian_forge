import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_keys_empty(self):
    """Test keys method"""
    model = ConcreteModel()
    model.o = Objective()
    self.assertEqual(list(model.o.keys()), [])