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
def create_model9():
    model = pyo.ConcreteModel()
    p = 71
    wght = -0.1
    hp2 = 0.5 * p ** 2
    model.x = pyo.Var(pyo.RangeSet(1, p), pyo.RangeSet(1, p), initialize=0.0)

    def f(model):
        return sum((0.5 * (model.x[i, j] - model.x[i, j - 1]) ** 2 + 0.5 * (model.x[i, j] - model.x[i - 1, j]) ** 2 + hp2 * (model.x[i, j] - model.x[i, j - 1]) ** 4 + hp2 * (model.x[i, j] - model.x[i - 1, j]) ** 4 for i in range(2, p + 1) for j in range(2, p + 1))) + wght * model.x[p, p]
    model.f = pyo.Objective(rule=f)
    for j in range(1, p + 1):
        model.x[1, j] = 0.0
        model.x[1, j].fixed = True
    return model