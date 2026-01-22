import os
import pyomo.common.unittest as unittest
from io import StringIO
import logging
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface
def simple_model_1():
    m = pyo.ConcreteModel()
    m.v1 = pyo.Var(initialize=10.0)
    m.v2 = pyo.Var(initialize=10.0)
    m.p = pyo.Param(mutable=True, initialize=1.0)
    m.eq_con = pyo.Constraint(expr=m.v1 * m.v2 - m.p == 0)
    m.obj = pyo.Objective(expr=m.v1 ** 2 + m.v2 ** 2, sense=pyo.minimize)
    return m