import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
@unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
def test_cyipopt_nlp(self):
    m = self.make_model()
    scaling_factors = [0.0001, 10000.0]
    m.epm.set_equality_constraint_scaling_factors(scaling_factors)
    nlp = PyomoNLPWithGreyBoxBlocks(m)
    cyipopt_nlp = CyIpoptNLP(nlp)
    obj_scaling, x_scaling, g_scaling = cyipopt_nlp.scaling_factors()
    np.testing.assert_array_equal(scaling_factors, g_scaling)