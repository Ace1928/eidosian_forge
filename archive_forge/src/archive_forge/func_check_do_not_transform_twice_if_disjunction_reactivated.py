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
def check_do_not_transform_twice_if_disjunction_reactivated(self, transformation):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    first_buf = StringIO()
    m.pprint(ostream=first_buf)
    first_output = first_buf.getvalue()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    second_buf = StringIO()
    m.pprint(ostream=second_buf)
    second_output = second_buf.getvalue()
    self.assertMultiLineEqual(first_output, second_output)
    m.disjunction.activate()
    self.assertRaisesRegex(GDP_Error, "The disjunct 'd\\[0\\]' has been transformed, but 'disjunction', a disjunction it appears in, has not. Putting the same disjunct in multiple disjunctions is not supported.", TransformationFactory('gdp.%s' % transformation).apply_to, m)