from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
def test_indexed_nested_disjunction(self):
    m = ConcreteModel()
    m.d1 = Disjunct()
    m.d1.indexedDisjunct1 = Disjunct([0, 1])
    m.d1.indexedDisjunct2 = Disjunct([0, 1])

    @m.d1.Disjunction([0, 1])
    def innerIndexed(d, i):
        return [d.indexedDisjunct1[i], d.indexedDisjunct2[i]]
    m.d2 = Disjunct()
    m.outer = Disjunction(expr=[m.d1, m.d2])
    TransformationFactory('gdp.bigm').apply_to(m)
    disjuncts = [m.d1, m.d2, m.d1.indexedDisjunct1[0], m.d1.indexedDisjunct1[1], m.d1.indexedDisjunct2[0], m.d1.indexedDisjunct2[1]]
    for disjunct in disjuncts:
        self.assertIs(disjunct.transformation_block.parent_component(), m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts)