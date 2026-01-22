import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_bounds_and_ineq_okay(self):
    m, _ = make_simple_model()
    m.x[1].setlb(0.0)
    m.x[1].setub(1.0)
    m.con4 = pyo.Constraint(expr=m.x[1] <= m.x[2])
    nlp = PyomoNLP(m)
    solver = SquareNlpSolverBase(nlp)