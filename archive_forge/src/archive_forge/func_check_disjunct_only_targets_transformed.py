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
def check_disjunct_only_targets_transformed(self, transformation):
    m = models.makeNestedDisjunctions()
    transform = TransformationFactory('gdp.%s' % transformation)
    transform.apply_to(m, targets=[m.simpledisjunct])
    disjBlock = m.simpledisjunct.component('_pyomo_gdp_%s_reformulation' % transformation).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 2)
    self.assertIs(transform.get_transformed_constraints(m.simpledisjunct.innerdisjunct0.c)[0].parent_block(), disjBlock[0])
    self.assertIs(transform.get_transformed_constraints(m.simpledisjunct.innerdisjunct0.c)[0].parent_block(), disjBlock[0])
    self.assertIs(transform.get_transformed_constraints(m.simpledisjunct.innerdisjunct1.c)[0].parent_block(), disjBlock[1])
    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(m.simpledisjunct.component('innerdisjunct%d' % i), transform.get_src_disjunct(disjBlock[j]))
        self.assertIs(disjBlock[j], m.simpledisjunct.component('innerdisjunct%d' % i).transformation_block)