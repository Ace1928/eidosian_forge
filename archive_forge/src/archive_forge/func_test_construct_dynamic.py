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
def test_construct_dynamic(self):
    m = make_dynamic_model()
    time = m.time
    t0 = m.time.first()
    inputs = [m.h, m.dhdt, m.flow_in]
    ext_vars = [m.flow_out]
    residuals = [m.h_diff_eqn]
    ext_cons = [m.flow_out_eqn]
    external_model_dict = {t: ExternalPyomoModel([var[t] for var in inputs], [var[t] for var in ext_vars], [con[t] for con in residuals], [con[t] for con in ext_cons]) for t in time}
    reduced_space = pyo.Block(concrete=True)
    reduced_space.external_block = ExternalGreyBoxBlock(time, external_model=external_model_dict)
    block = reduced_space.external_block
    block[t0].deactivate()
    self.assertIs(type(block), IndexedExternalGreyBoxBlock)
    for t in time:
        b = block[t]
        self.assertEqual(len(b.inputs), len(inputs))
        self.assertEqual(len(b.outputs), 0)
        self.assertEqual(len(b._equality_constraint_names), len(residuals))
    reduced_space.diff_var = pyo.Reference(m.h)
    reduced_space.deriv_var = pyo.Reference(m.dhdt)
    reduced_space.input_var = pyo.Reference(m.flow_in)
    reduced_space.disc_eqn = pyo.Reference(m.dhdt_disc_eqn)
    pyomo_vars = list(reduced_space.component_data_objects(pyo.Var))
    pyomo_cons = list(reduced_space.component_data_objects(pyo.Constraint))
    self.assertEqual(len(pyomo_vars), len(inputs) * len(time))
    self.assertEqual(len(pyomo_cons), len(time) - 1)
    reduced_space._obj = pyo.Objective(expr=0)
    block[:].inputs[:].set_value(1.0)
    reduced_space.const_input_eqn = pyo.Constraint(expr=reduced_space.input_var[2] - reduced_space.input_var[1] == 0)
    nlp = PyomoNLPWithGreyBoxBlocks(reduced_space)
    self.assertEqual(nlp.n_primals(), (2 + len(inputs)) * (len(time) - 1) + len(time))
    self.assertEqual(nlp.n_constraints(), (len(residuals) + 1) * (len(time) - 1) + 1)