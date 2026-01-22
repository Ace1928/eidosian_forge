import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_continuousSet(self):
    m = pyo.ConcreteModel(name='basicFormulation')
    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.v = pyo.Var(m.I)

    def ruleMaker(m):
        return sum((m.v[i] for i in m.I)) <= 0
    m.constraint = pyo.Constraint(rule=ruleMaker)
    pstr = latex_printer(m.constraint, explicit_set_summation=True)
    bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i = 1 }^{5} v_{i} \\leq 0 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)