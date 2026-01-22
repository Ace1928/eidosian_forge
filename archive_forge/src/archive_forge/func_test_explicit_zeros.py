import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
def test_explicit_zeros(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=0.0)
    m.eqn = pyo.Constraint(expr=m.x ** 2 + m.y ** 3 == 1.0)
    variables = [m.x, m.y]
    row = np.array([0, 1])
    col = np.array([0, 1])
    data = np.array([2.0, 0.0])
    expected_hess = sps.coo_matrix((data, (row, col)), shape=(2, 2))
    hess = get_hessian_of_constraint(m.eqn, variables)
    np.testing.assert_allclose(hess.row, row, atol=0)
    np.testing.assert_allclose(hess.col, col, atol=0)
    np.testing.assert_allclose(hess.data, data, rtol=1e-08)