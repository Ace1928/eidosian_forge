import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_solver_arg(self):
    m = ConcreteModel()
    m.x = Var()
    m.c = Constraint(expr=m.x == 2)
    m.o = Objective(expr=m.x)
    outs = StringIO()
    m.write(outs, format='gams', io_options=dict(solver='gurobi'))
    self.assertIn('option lp=gurobi', outs.getvalue())