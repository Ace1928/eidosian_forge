import pickle
from pyomo.common.dependencies import dill
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.base import constraint, ComponentUID
from pyomo.core.base.block import _BlockData
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
from io import StringIO
import random
import pyomo.opt
def check_disjunct_targets_inactive(self, transformation, **kwargs):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.simpledisjunct], **kwargs)
    self.assertTrue(m.disjunct.active)
    self.assertTrue(m.disjunct[0].active)
    self.assertTrue(m.disjunct[1].active)
    self.assertTrue(m.disjunct[1].innerdisjunct.active)
    self.assertTrue(m.disjunct[1].innerdisjunct[0].active)
    self.assertTrue(m.disjunct[1].innerdisjunct[1].active)
    self.assertTrue(m.simpledisjunct.active)
    self.assertFalse(m.simpledisjunct.innerdisjunct0.active)
    self.assertFalse(m.simpledisjunct.innerdisjunct1.active)