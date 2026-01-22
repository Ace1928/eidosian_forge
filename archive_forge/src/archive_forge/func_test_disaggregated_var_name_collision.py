from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def test_disaggregated_var_name_collision(self):
    m = ConcreteModel()
    x = Var(bounds=(2, 11))
    m.add_component('disj1.x', x)
    m.disj1 = Disjunct()
    m.disj1.x = Var(bounds=(1, 10))
    m.disj1.cons1 = Constraint(expr=m.disj1.x + x <= 5)
    m.disj2 = Disjunct()
    m.disj2.cons = Constraint(expr=x >= 8)
    m.disj2.cons1 = Constraint(expr=m.disj1.x == 3)
    m.disjunction1 = Disjunction(expr=[m.disj1, m.disj2])
    m.disj3 = Disjunct()
    m.disj3.cons = Constraint(expr=m.disj1.x >= 7)
    m.disj3.cons1 = Constraint(expr=x >= 10)
    m.disj4 = Disjunct()
    m.disj4.cons = Constraint(expr=x == 3)
    m.disj4.cons1 = Constraint(expr=m.disj1.x == 4)
    m.disjunction2 = Disjunction(expr=[m.disj3, m.disj4])
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    for disj in (m.disj1, m.disj2, m.disj3, m.disj4):
        self.check_name_collision_disaggregated_vars(m, disj)