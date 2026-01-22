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
def create_pyomo_external_grey_box_model(A1, A2, c1, c2, N, dt):
    m2 = pyo.ConcreteModel()
    m2.T = pyo.Set(initialize=list(range(N)), ordered=True)
    m2.Tu = pyo.Set(initialize=list(range(N))[1:], ordered=True)
    m2.egb = ExternalGreyBoxBlock()
    m2.egb.set_external_model(TwoTanksSeries(A1, A2, c1, c2, N, dt))
    for t in m2.Tu:
        m2.egb.inputs['F1_{}'.format(t)].value = 1 + 0.1 * t
        m2.egb.inputs['F2_{}'.format(t)].value = 2 + 0.1 * t
    for t in m2.T:
        m2.egb.inputs['h1_{}'.format(t)].value = 3 + 0.1 * t
        m2.egb.inputs['h2_{}'.format(t)].value = 4 + 0.1 * t
        m2.egb.outputs['F12_{}'.format(t)].value = 5 + 0.1 * t
        m2.egb.outputs['Fo_{}'.format(t)].value = 6 + 0.1 * t

    @m2.Constraint(m2.Tu)
    def min_inflow(m, t):
        F1_t = m.egb.inputs['F1_{}'.format(t)]
        return 2 <= F1_t

    @m2.Constraint(m2.T)
    def max_outflow(m, t):
        Fo_t = m.egb.outputs['Fo_{}'.format(t)]
        return Fo_t <= 4.5
    m2.h10 = pyo.Constraint(expr=m2.egb.inputs['h1_0'] == 1.5)
    m2.h20 = pyo.Constraint(expr=m2.egb.inputs['h2_0'] == 0.5)
    m2.obj = pyo.Objective(expr=sum(((m2.egb.inputs['h1_{}'.format(t)] - 1.0) ** 2 + (m2.egb.inputs['h2_{}'.format(t)] - 1.5) ** 2 for t in m2.T)))
    return m2