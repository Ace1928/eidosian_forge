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
def check_disjunction_data_target_any_index(self, transformation):
    m = ConcreteModel()
    m.x = Var(bounds=(-100, 100))
    m.disjunct3 = Disjunct(Any)
    m.disjunct4 = Disjunct(Any)
    m.disjunction2 = Disjunction(Any)
    for i in range(2):
        m.disjunct3[i].cons = Constraint(expr=m.x == 2)
        m.disjunct4[i].cons = Constraint(expr=m.x <= 3)
        m.disjunction2[i] = [m.disjunct3[i], m.disjunct4[i]]
        TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.disjunction2[i]])
        if i == 0:
            check_relaxation_block(self, m, '_pyomo_gdp_%s_reformulation' % transformation, 2)
        if i == 2:
            check_relaxation_block(self, m, '_pyomo_gdp_%s_reformulation' % transformation, 4)