import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
def test_pressure_drop_two_equalities_two_outputs(self):
    self._test_pressure_drop_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputs(), False)
    self._test_pressure_drop_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian(), True)