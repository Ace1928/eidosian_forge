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
def test_hierarchical_badly_ordered_targets(self):
    m = models.makeHierarchicalNested_DeclOrderMatchesInstantiationOrder()
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m, targets=[m.disjunction_block, m.disjunct_block.disj2])
    self.check_hierarchical_nested_model(m, bigm)