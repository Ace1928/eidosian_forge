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
def test_simple_model_1(self):
    model = SimpleModel1()
    m = model.make_model()
    m.x.set_value(2.0)
    m.y.set_value(2.0)
    con = m.residual_eqn
    expected_hess = np.array([[2.0, 0.0], [0.0, 2.0]])
    hess = get_hessian_of_constraint(con)
    self.assertTrue(np.all(expected_hess == hess.toarray()))
    expected_hess = np.array([[2.0]])
    hess = get_hessian_of_constraint(con, [m.x])
    self.assertTrue(np.all(expected_hess == hess.toarray()))
    con = m.external_eqn
    expected_hess = np.array([[0.0, 1.0], [1.0, 0.0]])
    hess = get_hessian_of_constraint(con)
    self.assertTrue(np.all(expected_hess == hess.toarray()))