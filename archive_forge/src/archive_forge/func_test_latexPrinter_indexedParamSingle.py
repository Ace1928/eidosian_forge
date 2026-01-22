import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_indexedParamSingle(self):
    m = pyo.ConcreteModel(name='basicFormulation')
    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.x = pyo.Var(m.I * m.I)
    m.c = pyo.Param(m.I * m.I, initialize=1.0, mutable=True)

    def ruleMaker_1(m):
        return sum((m.c[i, j] * m.x[i, j] for i in m.I for j in m.I))

    def ruleMaker_2(m):
        return sum((m.c[i, j] * m.x[i, j] ** 2 for i in m.I for j in m.I)) <= 1
    m.objective = pyo.Objective(rule=ruleMaker_1)
    m.constraint_1 = pyo.Constraint(rule=ruleMaker_2)
    pstr = latex_printer(m.constraint_1)
    print(pstr)
    bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i \\in I  } \\sum_{ j \\in I  } c_{i,j} x_{i,j}^{2} \\leq 1 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)