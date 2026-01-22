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
def check_block_only_targets_transformed(self, transformation):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=[m.b])
    disjBlock = m.b.component('_pyomo_gdp_%s_reformulation' % transformation).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 2)
    if transformation == 'bigm':
        self.assertIs(disjBlock[0], trans.get_transformed_constraints(m.b.disjunct[0].c)[0].parent_block())
    elif transformation == 'hull':
        self.assertIs(disjBlock[0], trans.get_transformed_constraints(m.b.disjunct[0].c)[0].parent_block().parent_block())
    self.assertIs(disjBlock[1], trans.get_transformed_constraints(m.b.disjunct[1].c)[0].parent_block())
    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(m.b.disjunct[i].transformation_block, disjBlock[j])
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.b.disjunct[i])