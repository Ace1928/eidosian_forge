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
def check_trans_block_created(self, transformation):
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    transBlock = m.b.component('_pyomo_gdp_%s_reformulation' % transformation)
    self.assertIsInstance(transBlock, Block)
    disjBlock = transBlock.component('relaxedDisjuncts')
    self.assertIsInstance(disjBlock, Block)
    self.assertEqual(len(disjBlock), 2)
    self.assertIsNone(m.component('_pyomo_gdp_%s_reformulation' % transformation))