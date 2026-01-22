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
def test_hessian_1(self):
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
    _add_nonlinear_linking_constraints(m)
    nlp = PyomoNLPWithGreyBoxBlocks(m)
    primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    duals = np.array([1, 1, 1, 1, 1])
    nlp.set_primals(primals)
    nlp.set_duals(duals)
    hess = nlp.evaluate_hessian_lag()
    row = [0, 1, 7]
    col = [0, 1, 7]
    data = [2.0, 2.0, 2.0]
    rcd_dict = dict((((i, j), val) for i, j, val in zip(row, col, data)))
    ex_block_nonzeros = {(2, 2): 2.0 + -1.0 + -0.10967928 + -0.25595929, (2, 3): -0.10684633 + 0.05169308, (3, 2): -0.10684633 + 0.05169308, (2, 4): 0.19329898 + 0.03823075, (4, 2): 0.19329898 + 0.03823075, (3, 3): 2.0 + -1.0 + -1.31592135 + -0.0241836, (3, 4): 1.13920361 + 0.01063667, (4, 3): 1.13920361 + 0.01063667, (4, 4): 2.0 + -1.0 + -1.0891866 + 0.01190218, (5, 5): 2.0, (6, 6): 2.0}
    rcd_dict.update(ex_block_nonzeros)
    ex_block_coords = [2, 3, 4, 5, 6]
    for i, j in itertools.product(ex_block_coords, ex_block_coords):
        row.append(i)
        col.append(j)
        if (i, j) not in rcd_dict:
            rcd_dict[i, j] = 0.0
    self.assertEqual(len(row), len(hess.row))
    for i, j, val in zip(hess.row, hess.col, hess.data):
        self.assertIn((i, j), rcd_dict)
        self.assertAlmostEqual(rcd_dict[i, j], val, delta=1e-08)