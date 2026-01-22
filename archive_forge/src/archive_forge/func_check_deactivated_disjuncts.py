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
def check_deactivated_disjuncts(self, transformation, **kwargs):
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,), **kwargs)
    for i in m.disjunct.index_set():
        self.assertFalse(m.disjunct[i].active)
    self.assertFalse(m.disjunct.active)