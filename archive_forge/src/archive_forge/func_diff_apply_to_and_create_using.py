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
def diff_apply_to_and_create_using(self, model, transformation, **kwargs):
    modelcopy = TransformationFactory(transformation).create_using(model, **kwargs)
    modelcopy_buf = StringIO()
    modelcopy.pprint(ostream=modelcopy_buf)
    modelcopy_output = modelcopy_buf.getvalue()
    random.seed(666)
    TransformationFactory(transformation).apply_to(model, **kwargs)
    model_buf = StringIO()
    model.pprint(ostream=model_buf)
    model_output = model_buf.getvalue()
    self.assertMultiLineEqual(modelcopy_output, model_output)