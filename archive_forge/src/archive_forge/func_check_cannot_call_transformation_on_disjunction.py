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
def check_cannot_call_transformation_on_disjunction(self, transformation, **kwargs):
    m = models.makeTwoTermIndexedDisjunction()
    trans = TransformationFactory('gdp.%s' % transformation)
    self.assertRaisesRegex(GDP_Error, "Transformation called on disjunction of type <class 'pyomo.gdp.disjunct.Disjunction'>. 'instance' must be a ConcreteModel, Block, or Disjunct \\(in the case of nested disjunctions\\).", trans.apply_to, m.disjunction, targets=m.disjunction[1], **kwargs)