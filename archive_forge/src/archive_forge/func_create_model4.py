import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
def create_model4():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=1.0)
    m.c1 = pyo.Constraint(expr=m.x[1] + m.x[2] - 1 == 0)
    m.obj = pyo.Objective(expr=2 * m.x[1] ** 2 + m.x[2] ** 2)
    return m