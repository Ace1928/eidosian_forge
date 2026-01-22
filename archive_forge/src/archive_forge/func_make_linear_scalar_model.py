import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def make_linear_scalar_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0, bounds=(0.0, None))
    m.con = pyo.Constraint(expr=-12.5 * m.x + 30.1 == 0)
    m.obj = pyo.Objective(expr=0.0)
    nlp = PyomoNLP(m)
    return (m, nlp)