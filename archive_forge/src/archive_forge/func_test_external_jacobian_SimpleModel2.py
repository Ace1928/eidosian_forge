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
def test_external_jacobian_SimpleModel2(self):
    model = SimpleModel2()
    m = model.make_model()
    x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
    external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
    for x in x_init_list:
        external_model.set_input_values(x)
        jac = external_model.evaluate_jacobian_external_variables()
        expected_jac = model.evaluate_external_jacobian(x[0])
        self.assertAlmostEqual(jac[0, 0], expected_jac, delta=1e-08)