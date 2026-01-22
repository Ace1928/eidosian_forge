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
def evaluate_external_hessian(self, x):
    d2y0dx0dx0 = 0.4 / (x[0] ** 3 * x[1])
    d2y0dx0dx1 = 0.2 / (x[0] ** 2 * x[1] ** 2)
    d2y0dx1dx1 = 0.4 / (x[0] * x[1] ** 3)
    d2y1dx0dx0 = x[1] ** 0.5 / x[0] ** 3
    d2y1dx0dx1 = -0.25 / (x[0] ** 2 * x[1] ** 0.5)
    d2y1dx1dx1 = -0.125 / (x[0] * x[1] ** 1.5)
    d2y0dxdx = np.array([[d2y0dx0dx0, d2y0dx0dx1], [d2y0dx0dx1, d2y0dx1dx1]])
    d2y1dxdx = np.array([[d2y1dx0dx0, d2y1dx0dx1], [d2y1dx0dx1, d2y1dx1dx1]])
    return [d2y0dxdx, d2y1dxdx]