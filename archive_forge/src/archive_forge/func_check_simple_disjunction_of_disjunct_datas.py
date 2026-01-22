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
def check_simple_disjunction_of_disjunct_datas(self, transformation):
    m = models.makeDisjunctionOfDisjunctDatas()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    self.check_trans_block_disjunctions_of_disjunct_datas(m)
    transBlock = m.component('_pyomo_gdp_%s_reformulation' % transformation)
    self.assertIsInstance(transBlock.component('disjunction_xor'), Constraint)
    self.assertIsInstance(transBlock.component('disjunction2_xor'), Constraint)