import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
@unittest.skipIf(not cyipopt_available or cyipopt_ge_1_3, 'cyipopt version >= 1.3.0')
def test_hs071_evalerror_old_cyipopt(self):
    m = make_hs071_model()
    solver = pyo.SolverFactory('cyipopt')
    msg = 'Error in AMPL evaluation'
    with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
        res = solver.solve(m, tee=True)