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
def check_deactivated_constraints(self, transformation, **kwargs):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, **kwargs)
    oldblock = m.component('d')
    oldc1 = oldblock[1].component('c1')
    self.assertIsInstance(oldc1, Constraint)
    self.assertFalse(oldc1.active)
    oldc2 = oldblock[1].component('c2')
    self.assertIsInstance(oldc2, Constraint)
    self.assertFalse(oldc2.active)
    oldc = oldblock[0].component('c')
    self.assertIsInstance(oldc, Constraint)
    self.assertFalse(oldc.active)