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

    def test_pressure_drop_two_equalities_two_outputs_no_hessian(self):
        #   u = [Pin, c, F, P1, P3]
        #   o = {P2, Pout}
        #   h_eq(u) = [P1 - (Pin - c*F^2]
        #             [P3 - (Pin - 2*c*F^2]
        #   h_o(u) = [P1 - c*F^2]
        #            [Pin - 4*c*F^2]
        egbm = ex_models.PressureDropTwoEqualitiesTwoOutputsNoHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P1', 'P3'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop1', 'pdrop3'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3, 80, 70], dtype=np.float64))
        with self.assertRaises(NotImplementedError):
            egbm.set_equality_constraint_multipliers(np.asarray([2, 4], dtype=np.float64))
        with self.assertRaises(NotImplementedError):
            egbm.set_output_constraint_multipliers(np.asarray([7, 9], dtype=np.float64))
            
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-2, 26], dtype=np.float64)))

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([62, 28], dtype=np.float64)))

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0,0,0,0,1,1,1,1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0,1,2,3,1,2,3,4], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 9, 12, 1, 18, 24, -1, 1], dtype=np.float64)))

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0,0,0,1,1,1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([1,2,3,0,1,2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([-9, -12, 1, 1, -36, -48], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            hess = egbm.evaluate_hessian_equality_constraints()

        with self.assertRaises(NotImplementedError):
            hess = egbm.evaluate_hessian_outputs()
