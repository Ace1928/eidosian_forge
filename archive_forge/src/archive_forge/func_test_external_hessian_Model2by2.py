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
def test_external_hessian_Model2by2(self):
    model = Model2by2()
    m = model.make_model()
    m.x[0].set_value(1.0)
    m.x[1].set_value(2.0)
    m.y[0].set_value(3.0)
    m.y[1].set_value(4.0)
    x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
    x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
    x_init_list = list(itertools.product(x0_init_list, x1_init_list))
    external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
    for x in x_init_list:
        external_model.set_input_values(x)
        hess = external_model.evaluate_hessian_external_variables()
        expected_hess = model.evaluate_external_hessian(x)
        for matrix1, matrix2 in zip(hess, expected_hess):
            np.testing.assert_allclose(matrix1, matrix2, rtol=1e-08)