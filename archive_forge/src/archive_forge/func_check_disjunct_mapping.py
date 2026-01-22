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
def check_disjunct_mapping(self, transformation):
    m = models.makeTwoTermDisj_Nonlinear()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)
    disjBlock = m.component('_pyomo_gdp_%s_reformulation' % transformation).relaxedDisjuncts
    for i in [0, 1]:
        self.assertIs(disjBlock[i]._src_disjunct(), m.d[i])
        self.assertIs(trans.get_src_disjunct(disjBlock[i]), m.d[i])