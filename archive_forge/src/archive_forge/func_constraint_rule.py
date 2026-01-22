import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Var, Constraint, value
def constraint_rule(m):
    return (m.Q, m.X, m.P)