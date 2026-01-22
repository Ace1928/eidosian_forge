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
def check_do_not_transform_deactivated_targets(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    m.b[1].deactivate()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b[0], m.b[1]])
    checkb0TargetsInactive(self, m)
    checkb0TargetsTransformed(self, m, transformation)