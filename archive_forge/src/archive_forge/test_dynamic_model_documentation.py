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

        self._input_names = ['F1_{}'.format(t) for t in range(1,N)]
        self._input_names.extend(['F2_{}'.format(t) for t in range(1,N)])
        self._input_names.extend(['h1_{}'.format(t) for t in range(0,N)])
        self._input_names.extend(['h2_{}'.format(t) for t in range(0,N)])
        self._output_names = ['F12_{}'.format(t) for t in range(0,N)]
        self._output_names.extend(['Fo_{}'.format(t) for t in range(0,N)])
        