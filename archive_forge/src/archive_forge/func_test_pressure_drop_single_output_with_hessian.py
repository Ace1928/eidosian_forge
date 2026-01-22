import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
from ..pyomo_nlp import PyomoGreyBoxNLP
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
def test_pressure_drop_single_output_with_hessian(self):
    egbm = ex_models.PressureDropSingleOutputWithHessian()
    input_names = egbm.input_names()
    self.assertEqual(input_names, ['Pin', 'c', 'F'])
    eq_con_names = egbm.equality_constraint_names()
    self.assertEqual(eq_con_names, [])
    output_names = egbm.output_names()
    self.assertEqual(output_names, ['Pout'])
    egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
    egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
    with self.assertRaises(AssertionError):
        egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))
    egbm.set_output_constraint_multipliers(np.asarray([5], dtype=np.float64))
    with self.assertRaises(NotImplementedError):
        tmp = egbm.evaluate_equality_constraints()
    o = egbm.evaluate_outputs()
    self.assertTrue(np.array_equal(o, np.asarray([28], dtype=np.float64)))
    with self.assertRaises(NotImplementedError):
        tmp = egbm.evaluate_jacobian_equality_constraints()
    jac_o = egbm.evaluate_jacobian_outputs()
    self.assertTrue(np.array_equal(jac_o.row, np.asarray([0, 0, 0], dtype=np.int64)))
    self.assertTrue(np.array_equal(jac_o.col, np.asarray([0, 1, 2], dtype=np.int64)))
    self.assertTrue(np.array_equal(jac_o.data, np.asarray([1, -36, -48], dtype=np.float64)))
    with self.assertRaises(AttributeError):
        eq_hess = egbm.evaluate_hessian_equality_constraints()
    outputs_hess = egbm.evaluate_hessian_outputs()
    self.assertTrue(np.array_equal(outputs_hess.row, np.asarray([2, 2], dtype=np.int64)))
    self.assertTrue(np.array_equal(outputs_hess.col, np.asarray([1, 2], dtype=np.int64)))
    self.assertTrue(np.array_equal(outputs_hess.data, np.asarray([5 * (-8 * 3), 5 * (-8 * 2)], dtype=np.int64)))