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
@unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
def test_solve_square_dynamic(self):
    m = make_dynamic_model()
    time = m.time
    t0 = m.time.first()
    m.h[t0].fix(1.2)
    m.flow_in.fix(1.5)
    reduced_space = pyo.Block(concrete=True)
    reduced_space.diff_var = pyo.Reference(m.h)
    reduced_space.deriv_var = pyo.Reference(m.dhdt)
    reduced_space.input_var = pyo.Reference(m.flow_in)
    reduced_space.disc_eq = pyo.Reference(m.dhdt_disc_eqn)
    reduced_space.external_block = ExternalGreyBoxBlock(time)
    block = reduced_space.external_block
    block[t0].deactivate()
    for t in time:
        if t != t0:
            input_vars = [m.h[t], m.dhdt[t]]
            external_vars = [m.flow_out[t]]
            residual_cons = [m.h_diff_eqn[t]]
            external_cons = [m.flow_out_eqn[t]]
            external_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
            block[t].set_external_model(external_model)
    n_inputs = len(input_vars)

    def linking_constraint_rule(m, i, t):
        if t == t0:
            return pyo.Constraint.Skip
        if i == 0:
            return m.diff_var[t] == m.external_block[t].inputs['input_0']
        elif i == 1:
            return m.deriv_var[t] == m.external_block[t].inputs['input_1']
    reduced_space.linking_constraint = pyo.Constraint(range(n_inputs), time, rule=linking_constraint_rule)
    for t in time:
        if t != t0:
            block[t].inputs['input_0'].set_value(m.h[t].value)
            block[t].inputs['input_1'].set_value(m.dhdt[t].value)
    reduced_space._obj = pyo.Objective(expr=0)
    solver = pyo.SolverFactory('cyipopt')
    results = solver.solve(reduced_space, tee=True)
    h_target = [1.2, 0.852923, 0.690725]
    dhdt_target = [-0.69089, -0.347077, -0.162198]
    flow_out_target = [2.19098, 1.847077, 1.662198]
    for t in time:
        if t == t0:
            continue
        values = [m.h[t].value, m.dhdt[t].value, m.flow_out[t].value]
        target_values = [h_target[t], dhdt_target[t], flow_out_target[t]]
        self.assertStructuredAlmostEqual(values, target_values, delta=1e-05)