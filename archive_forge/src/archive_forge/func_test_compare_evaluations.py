import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import math
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoGreyBoxNLP, PyomoNLP
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
@unittest.skipIf(not pyo.SolverFactory('ipopt').available(exception_flag=False), 'Ipopt needed to run tests with solve')
def test_compare_evaluations(self):
    A1 = 5
    A2 = 10
    c1 = 3
    c2 = 4
    N = 6
    dt = 1
    m = create_pyomo_model(A1, A2, c1, c2, N, dt)
    solver = pyo.SolverFactory('ipopt')
    solver.options['linear_solver'] = 'mumps'
    status = solver.solve(m, tee=False)
    m_nlp = PyomoNLP(m)
    mex = create_pyomo_external_grey_box_model(A1, A2, c1, c2, N, dt)
    mex_nlp = PyomoNLPWithGreyBoxBlocks(mex)
    m_x_order = m_nlp.primals_names()
    m_c_order = m_nlp.constraint_names()
    mex_x_order = mex_nlp.primals_names()
    mex_c_order = mex_nlp.constraint_names()
    x1list = ['h1[0]', 'h1[1]', 'h1[2]', 'h1[3]', 'h1[4]', 'h1[5]', 'h2[0]', 'h2[1]', 'h2[2]', 'h2[3]', 'h2[4]', 'h2[5]', 'F1[1]', 'F1[2]', 'F1[3]', 'F1[4]', 'F1[5]', 'F2[1]', 'F2[2]', 'F2[3]', 'F2[4]', 'F2[5]', 'F12[0]', 'F12[1]', 'F12[2]', 'F12[3]', 'F12[4]', 'F12[5]', 'Fo[0]', 'Fo[1]', 'Fo[2]', 'Fo[3]', 'Fo[4]', 'Fo[5]']
    x2list = ['egb.inputs[h1_0]', 'egb.inputs[h1_1]', 'egb.inputs[h1_2]', 'egb.inputs[h1_3]', 'egb.inputs[h1_4]', 'egb.inputs[h1_5]', 'egb.inputs[h2_0]', 'egb.inputs[h2_1]', 'egb.inputs[h2_2]', 'egb.inputs[h2_3]', 'egb.inputs[h2_4]', 'egb.inputs[h2_5]', 'egb.inputs[F1_1]', 'egb.inputs[F1_2]', 'egb.inputs[F1_3]', 'egb.inputs[F1_4]', 'egb.inputs[F1_5]', 'egb.inputs[F2_1]', 'egb.inputs[F2_2]', 'egb.inputs[F2_3]', 'egb.inputs[F2_4]', 'egb.inputs[F2_5]', 'egb.outputs[F12_0]', 'egb.outputs[F12_1]', 'egb.outputs[F12_2]', 'egb.outputs[F12_3]', 'egb.outputs[F12_4]', 'egb.outputs[F12_5]', 'egb.outputs[Fo_0]', 'egb.outputs[Fo_1]', 'egb.outputs[Fo_2]', 'egb.outputs[Fo_3]', 'egb.outputs[Fo_4]', 'egb.outputs[Fo_5]']
    x1_x2_map = dict(zip(x1list, x2list))
    x1idx_x2idx_map = {i: mex_x_order.index(x1_x2_map[m_x_order[i]]) for i in range(len(m_x_order))}
    c1list = ['h1bal[1]', 'h1bal[2]', 'h1bal[3]', 'h1bal[4]', 'h1bal[5]', 'h2bal[1]', 'h2bal[2]', 'h2bal[3]', 'h2bal[4]', 'h2bal[5]', 'F12con[0]', 'F12con[1]', 'F12con[2]', 'F12con[3]', 'F12con[4]', 'F12con[5]', 'Focon[0]', 'Focon[1]', 'Focon[2]', 'Focon[3]', 'Focon[4]', 'Focon[5]', 'min_inflow[1]', 'min_inflow[2]', 'min_inflow[3]', 'min_inflow[4]', 'min_inflow[5]', 'max_outflow[0]', 'max_outflow[1]', 'max_outflow[2]', 'max_outflow[3]', 'max_outflow[4]', 'max_outflow[5]', 'h10', 'h20']
    c2list = ['egb.h1bal_1', 'egb.h1bal_2', 'egb.h1bal_3', 'egb.h1bal_4', 'egb.h1bal_5', 'egb.h2bal_1', 'egb.h2bal_2', 'egb.h2bal_3', 'egb.h2bal_4', 'egb.h2bal_5', 'egb.output_constraints[F12_0]', 'egb.output_constraints[F12_1]', 'egb.output_constraints[F12_2]', 'egb.output_constraints[F12_3]', 'egb.output_constraints[F12_4]', 'egb.output_constraints[F12_5]', 'egb.output_constraints[Fo_0]', 'egb.output_constraints[Fo_1]', 'egb.output_constraints[Fo_2]', 'egb.output_constraints[Fo_3]', 'egb.output_constraints[Fo_4]', 'egb.output_constraints[Fo_5]', 'min_inflow[1]', 'min_inflow[2]', 'min_inflow[3]', 'min_inflow[4]', 'min_inflow[5]', 'max_outflow[0]', 'max_outflow[1]', 'max_outflow[2]', 'max_outflow[3]', 'max_outflow[4]', 'max_outflow[5]', 'h10', 'h20']
    c1_c2_map = dict(zip(c1list, c2list))
    c1idx_c2idx_map = {i: mex_c_order.index(c1_c2_map[m_c_order[i]]) for i in range(len(m_c_order))}
    m_x = m_nlp.get_primals()
    mex_x = np.zeros(len(m_x))
    for i in range(len(m_x)):
        mex_x[x1idx_x2idx_map[i]] = m_x[i]
    m_lam = m_nlp.get_duals()
    mex_lam = np.zeros(len(m_lam))
    for i in range(len(m_x)):
        mex_lam[c1idx_c2idx_map[i]] = m_lam[i]
    mex_nlp.set_primals(mex_x)
    mex_nlp.set_duals(mex_lam)
    m_obj = m_nlp.evaluate_objective()
    mex_obj = mex_nlp.evaluate_objective()
    self.assertAlmostEqual(m_obj, mex_obj, places=4)
    m_gobj = m_nlp.evaluate_grad_objective()
    mex_gobj = mex_nlp.evaluate_grad_objective()
    check_vectors_specific_order(self, m_gobj, m_x_order, mex_gobj, mex_x_order, x1_x2_map)
    m_c = m_nlp.evaluate_constraints()
    mex_c = mex_nlp.evaluate_constraints()
    check_vectors_specific_order(self, m_c, m_c_order, mex_c, mex_c_order, c1_c2_map)
    m_j = m_nlp.evaluate_jacobian()
    mex_j = mex_nlp.evaluate_jacobian().todense()
    check_sparse_matrix_specific_order(self, m_j, m_c_order, m_x_order, mex_j, mex_c_order, mex_x_order, c1_c2_map, x1_x2_map)
    m_h = m_nlp.evaluate_hessian_lag()
    mex_h = mex_nlp.evaluate_hessian_lag()
    check_sparse_matrix_specific_order(self, m_h, m_x_order, m_x_order, mex_h, mex_x_order, mex_x_order, x1_x2_map, x1_x2_map)
    mex_h = 0 * mex_h
    mex_nlp.evaluate_hessian_lag(out=mex_h)
    check_sparse_matrix_specific_order(self, m_h, m_x_order, m_x_order, mex_h, mex_x_order, mex_x_order, x1_x2_map, x1_x2_map)