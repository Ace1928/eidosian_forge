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
def test_pressure_drop_model(self):
    m = self._create_pressure_drop_model()
    cons = [m.c_con, m.F_con, m.Pin_con, m.P2_con]
    inputs = [m.Pin, m.c, m.F]
    outputs = [m.P2, m.Pout]
    pyomo_variables = list(m.component_data_objects(pyo.Var))
    pyomo_constraints = list(m.component_data_objects(pyo.Constraint))
    self.assertEqual(len(pyomo_variables), len(inputs) + len(outputs))
    self.assertEqual(len(pyomo_constraints), len(cons))
    self.assertIs(m.egb.inputs.ctype, pyo.Var)
    self.assertIs(m.egb.outputs.ctype, pyo.Var)
    self.assertEqual(len(m.egb.inputs), len(inputs))
    self.assertEqual(len(m.egb.outputs), len(outputs))
    for i in range(len(inputs)):
        self.assertIs(inputs[i], m.egb.inputs[i])
    for i in range(len(outputs)):
        self.assertIs(outputs[i], m.egb.outputs[i])