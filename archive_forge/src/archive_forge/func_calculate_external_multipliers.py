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
def calculate_external_multipliers(self, lam, x):
    """
        Calculates the multipliers of the external constraints
        from the multipliers of the residual constraints,
        assuming zero dual infeasibility in the coordinates of
        the external variables.
        This is calculated analytically from:

        \\nabla_y f^T \\lambda_f + \\nabla_y g^T \\lambda_g = 0

        """
    y = self.evaluate_external_variables(x)
    lg0 = -2 * y[1] * lam[0] / (x[0] ** 2 * x[1] ** 0.5 * y[0])
    lg1 = -(2 * y[0] * lam[0] + x[0] ** 2 * x[1] ** 0.5 * y[1] * lg0) / (x[0] * x[1])
    return [lg0, lg1]