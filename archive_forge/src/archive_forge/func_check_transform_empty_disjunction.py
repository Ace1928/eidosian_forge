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
def check_transform_empty_disjunction(self, transformation, **kwargs):
    m = ConcreteModel()
    m.empty = Disjunction(expr=[])
    self.assertRaisesRegex(GDP_Error, "Disjunction 'empty' is empty. This is likely indicative of a modeling error.*", TransformationFactory('gdp.%s' % transformation).apply_to, m, **kwargs)