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
def check_xor_constraint_mapping(self, transformation):
    m = models.makeTwoTermDisj()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)
    transBlock = m.component('_pyomo_gdp_%s_reformulation' % transformation)
    self.assertIs(trans.get_src_disjunction(transBlock.disjunction_xor), m.disjunction)
    self.assertIs(m.disjunction.algebraic_constraint, transBlock.disjunction_xor)