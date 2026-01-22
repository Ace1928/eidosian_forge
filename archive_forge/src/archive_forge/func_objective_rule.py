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
def objective_rule(m):
    accum = 0.0
    for i in m.var_ids:
        accum += m.x[i] * sum((m.hessian_f[i, j] * m.x[j] for j in m.var_ids))
    accum *= 0.5
    accum += sum((m.x[j] * m.grad_f[j] for j in m.var_ids))
    return accum