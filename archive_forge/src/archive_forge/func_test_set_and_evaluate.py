import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
def test_set_and_evaluate(self):
    m = pyo.ConcreteModel()
    m.ex_block = ExternalGreyBoxBlock(concrete=True)
    block = m.ex_block
    m_ex = _make_external_model()
    input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
    external_vars = [m_ex.x, m_ex.y]
    residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
    external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
    ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
    block.set_external_model(ex_model)
    a = m.ex_block.inputs['input_0']
    b = m.ex_block.inputs['input_1']
    r = m.ex_block.inputs['input_2']
    x = m.ex_block.inputs['input_3']
    y = m.ex_block.inputs['input_4']
    m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
    _add_linking_constraints(m)
    nlp = PyomoNLPWithGreyBoxBlocks(m)
    self.assertEqual(nlp.n_primals(), 8)
    primals_names = ['a', 'b', 'ex_block.inputs[input_0]', 'ex_block.inputs[input_1]', 'ex_block.inputs[input_2]', 'ex_block.inputs[input_3]', 'ex_block.inputs[input_4]', 'r']
    self.assertEqual(nlp.primals_names(), primals_names)
    np.testing.assert_equal(np.zeros(8), nlp.get_primals())
    primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    nlp.set_primals(primals)
    np.testing.assert_equal(primals, nlp.get_primals())
    nlp.load_state_into_pyomo()
    for name, val in zip(primals_names, primals):
        var = m.find_component(name)
        self.assertEqual(var.value, val)
    constraint_names = ['linking_constraint[0]', 'linking_constraint[1]', 'linking_constraint[2]', 'ex_block.residual_0', 'ex_block.residual_1']
    self.assertEqual(constraint_names, nlp.constraint_names())
    residuals = np.array([-2.0, -2.0, 3.0, 5.0 - -3.03051522, 6.0 - 3.583839997])
    np.testing.assert_allclose(residuals, nlp.evaluate_constraints(), rtol=1e-08)
    duals = np.array([1, 2, 3, 4, 5])
    nlp.set_duals(duals)
    self.assertEqual(ex_model.residual_con_multipliers, [4, 5])
    np.testing.assert_equal(nlp.get_duals(), duals)