import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
def test_var_domain_setter(self):
    m = ConcreteModel()
    m.x = Var([1, 2, 3])
    self.assertIs(m.x[1].domain, Reals)
    self.assertIs(m.x[2].domain, Reals)
    self.assertIs(m.x[3].domain, Reals)
    m.x.domain = Integers
    self.assertIs(m.x[1].domain, Integers)
    self.assertIs(m.x[2].domain, Integers)
    self.assertIs(m.x[3].domain, Integers)
    m.x.domain = lambda m, i: PositiveReals
    self.assertIs(m.x[1].domain, PositiveReals)
    self.assertIs(m.x[2].domain, PositiveReals)
    self.assertIs(m.x[3].domain, PositiveReals)
    m.x.domain = {1: Reals, 2: NonPositiveReals, 3: NonNegativeReals}
    self.assertIs(m.x[1].domain, Reals)
    self.assertIs(m.x[2].domain, NonPositiveReals)
    self.assertIs(m.x[3].domain, NonNegativeReals)
    m.x.domain = {2: Integers}
    self.assertIs(m.x[1].domain, Reals)
    self.assertIs(m.x[2].domain, Integers)
    self.assertIs(m.x[3].domain, NonNegativeReals)
    with LoggingIntercept() as LOG, self.assertRaisesRegex(TypeError, "Cannot create a Set from data that does not support __contains__.  Expected set-like object supporting collections.abc.Collection interface, but received 'NoneType'"):
        m.x.domain = {1: None, 2: None, 3: None}
    self.assertIn('{1: None, 2: None, 3: None} is not a valid domain. Variable domains must be an instance of a Pyomo Set or convertible to a Pyomo Set.', LOG.getvalue())