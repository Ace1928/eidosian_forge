import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, SOSConstraint, Var, Set
def test_arg1(self):
    M = ConcreteModel()
    try:
        M.c = SOSConstraint()
        self.fail('Expected TypeError')
    except TypeError:
        pass