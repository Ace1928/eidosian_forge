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
def check_error_for_same_disjunct_in_multiple_disjunctions(self, transformation, **kwargs):
    m = models.makeDisjunctInMultipleDisjunctions()
    self.assertRaisesRegex(GDP_Error, "The disjunct 'disjunct1\\[1\\]' has been transformed, but 'disjunction2', a disjunction it appears in, has not. Putting the same disjunct in multiple disjunctions is not supported.", TransformationFactory('gdp.%s' % transformation).apply_to, m, **kwargs)