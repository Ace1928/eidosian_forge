import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_simpleDocTests(self):
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var()
    m.y = pyo.Var()
    pstr = latex_printer(m.x + m.y)
    bstr = dedent('\n        \\begin{equation} \n             x + y \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.expression_1 = pyo.Expression(expr=m.x ** 2 + m.y ** 2)
    pstr = latex_printer(m.expression_1)
    bstr = dedent('\n        \\begin{equation} \n             x^{2} + y^{2} \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.constraint_1 = pyo.Constraint(expr=m.x ** 2 + m.y ** 2 <= 1.0)
    pstr = latex_printer(m.constraint_1)
    bstr = dedent('\n        \\begin{equation} \n             x^{2} + y^{2} \\leq 1 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    m = pyo.ConcreteModel(name='basicFormulation')
    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.v = pyo.Var(m.I)

    def ruleMaker(m):
        return sum((m.v[i] for i in m.I)) <= 0
    m.constraint = pyo.Constraint(rule=ruleMaker)
    pstr = latex_printer(m.constraint)
    bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i \\in I  } v_{i} \\leq 0 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    m.c = pyo.Param(initialize=1.0, mutable=True)
    m.objective = pyo.Objective(expr=m.x + m.y + m.z)
    m.constraint_1 = pyo.Constraint(expr=m.x ** 2 + m.y ** 2.0 - m.z ** 2.0 <= m.c)
    pstr = latex_printer(m)
    bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & x + y + z & \\label{obj:basicFormulation_objective} \\\\ \n            & \\text{s.t.} \n            & & x^{2} + y^{2} - z^{2} \\leq c & \\label{con:basicFormulation_constraint_1} \n        \\end{align} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    m = pyo.ConcreteModel(name='basicFormulation')
    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.v = pyo.Var(m.I)

    def ruleMaker(m):
        return sum((m.v[i] for i in m.I)) <= 0
    m.constraint = pyo.Constraint(rule=ruleMaker)
    lcm = ComponentMap()
    lcm[m.v] = 'x'
    lcm[m.I] = ['\\mathcal{A}', ['j', 'k']]
    pstr = latex_printer(m.constraint, latex_component_map=lcm)
    bstr = dedent('\n        \\begin{equation} \n             \\sum_{ j \\in \\mathcal{A}  } x_{j} \\leq 0 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)