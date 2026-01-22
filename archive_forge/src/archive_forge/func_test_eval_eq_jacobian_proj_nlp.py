import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def test_eval_eq_jacobian_proj_nlp(self):
    m, nlp, proj_nlp = self._get_nlps()
    x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
    nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))
    jac = proj_nlp.evaluate_jacobian_eq()
    self.assertEqual(jac.shape, (2, 2))
    pred_rc = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pred_data_dict = {(0, 0): x1 ** 1.1 * x2 ** 1.2, (0, 1): 1.1 * x0 * x1 ** 0.1 * x2 ** 1.2, (1, 0): 2 * x0, (1, 1): 1.0}
    pred_rc_set = set((self._rc_to_proj_nlp_eq(m, nlp, rc) for rc in pred_rc))
    pred_data_dict = {self._rc_to_proj_nlp_eq(m, nlp, rc): val for rc, val in pred_data_dict.items()}
    rc_set = set(zip(jac.row, jac.col))
    self.assertEqual(pred_rc_set, rc_set)
    data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
    self.assertEqual(pred_data_dict, data_dict)