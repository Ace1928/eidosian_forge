import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
def make_model_2():
    m = ConcreteModel()
    m.x = Var(initialize=0.1, bounds=(0, 1))
    m.y = Var(initialize=0.1, bounds=(0, 1))
    m.obj = Objective(expr=-m.x ** 2 - m.y ** 2)
    m.c = Constraint(expr=m.y <= pyo.exp(-m.x))
    return m