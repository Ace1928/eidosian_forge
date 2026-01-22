import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP, AmplNLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import tempfile
from pyomo.contrib.pynumero.interfaces.utils import (
def test_no_objective(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c = pyo.Constraint(expr=2.0 * m.x >= 5)
    with self.assertRaises(NotImplementedError):
        nlp = PyomoNLP(m)