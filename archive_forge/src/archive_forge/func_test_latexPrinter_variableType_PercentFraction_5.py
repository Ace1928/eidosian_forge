import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
from pyomo.common.errors import InfeasibleConstraintException
def test_latexPrinter_variableType_PercentFraction_5(self):
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var(domain=PercentFraction, bounds=(-10, -2))
    m.objective = pyo.Objective(expr=m.x)
    m.constraint_1 = pyo.Constraint(expr=m.x ** 2 <= 5.0)
    self.assertRaises(InfeasibleConstraintException, latex_printer, **{'pyomo_component': m})