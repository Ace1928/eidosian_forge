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
def test_model3(self):
    G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
    A = np.array([[1, 0, 1], [0, 1, 1]])
    b = np.array([3, 0])
    c = np.array([-8, -3, -3])
    model = create_model3(G, A, b, c)
    nlp = PyomoNLP(model)
    solver = CyIpoptSolver(CyIpoptNLP(nlp))
    x, info = solver.solve(tee=False)
    x_sol = np.array([2.0, -1.0, 1.0])
    y_sol = np.array([-3.0, 2.0])
    self.assertTrue(np.allclose(x, x_sol, rtol=0.0001))
    nlp.set_primals(x)
    nlp.set_duals(y_sol)
    self.assertAlmostEqual(nlp.evaluate_objective(), -3.5, places=5)
    self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=0.0001))