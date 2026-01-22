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
def make_dynamic_model():
    m = pyo.ConcreteModel()
    m.time = pyo.Set(initialize=[0, 1, 2])
    m = pyo.ConcreteModel()
    m.time = pyo.Set(initialize=[0, 1, 2])
    t0 = m.time.first()
    m.h = pyo.Var(m.time, initialize=1.0)
    m.dhdt = pyo.Var(m.time, initialize=1.0)
    m.flow_in = pyo.Var(m.time, bounds=(0, None), initialize=1.0)
    m.flow_out = pyo.Var(m.time, initialize=1.0)
    m.flow_coef = pyo.Param(initialize=2.0, mutable=True)

    def h_diff_eqn_rule(m, t):
        return m.dhdt[t] - (m.flow_in[t] - m.flow_out[t]) == 0
    m.h_diff_eqn = pyo.Constraint(m.time, rule=h_diff_eqn_rule)

    def dhdt_disc_eqn_rule(m, t):
        if t == m.time.first():
            return pyo.Constraint.Skip
        else:
            t_prev = m.time.prev(t)
            delta_t = t - t_prev
            return m.dhdt[t] - delta_t * (m.h[t] - m.h[t_prev]) == 0
    m.dhdt_disc_eqn = pyo.Constraint(m.time, rule=dhdt_disc_eqn_rule)

    def flow_out_eqn(m, t):
        return m.flow_out[t] == m.flow_coef * m.h[t] ** 0.5
    m.flow_out_eqn = pyo.Constraint(m.time, rule=flow_out_eqn)
    return m