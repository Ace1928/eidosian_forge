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
def test_transformed_model_nestedDisjuncts(self):
    m = models.makeNestedDisjunctions_NestedDisjuncts()
    m.LocalVars = Suffix(direction=Suffix.LOCAL)
    m.LocalVars[m.d1] = [m.d1.binary_indicator_var, m.d1.d3.binary_indicator_var, m.d1.d4.binary_indicator_var]
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    self.check_transformed_model_nestedDisjuncts(m, m.d1.d3.binary_indicator_var, m.d1.d4.binary_indicator_var)
    all_cons = list(m.component_data_objects(Constraint, active=True, descend_into=Block))
    self.assertEqual(len(all_cons), 16)