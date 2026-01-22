import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_overwriteError(self):
    m = pyo.ConcreteModel(name='basicFormulation')
    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.v = pyo.Var(m.I)

    def ruleMaker(m):
        return sum((m.v[i] for i in m.I)) <= 0
    m.constraint = pyo.Constraint(rule=ruleMaker)
    lcm = ComponentMap()
    lcm[m.v] = 'x'
    lcm[m.I] = ['\\mathcal{A}', ['j', 'k']]
    lcm['err'] = 1.0
    self.assertRaises(ValueError, latex_printer, **{'pyomo_component': m.constraint, 'latex_component_map': lcm})