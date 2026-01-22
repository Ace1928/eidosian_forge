import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def make_simple_model():
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[1, 2, 3])
    m.x = pyo.Var(m.I, initialize=1.0)
    m.con1 = pyo.Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 2 == 1)
    m.con2 = pyo.Constraint(expr=2 * m.x[1] + 3 * m.x[2] - 4 * m.x[3] == 0)
    m.con3 = pyo.Constraint(expr=m.x[1] == 2 * pyo.exp(m.x[2] / m.x[3]))
    m.obj = pyo.Objective(expr=0.0)
    nlp = PyomoNLP(m)
    return (m, nlp)